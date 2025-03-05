import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import soundfile as sf
import librosa
import scipy.signal
from datetime import timedelta
import srt
from tqdm import tqdm
from transformers import pipeline
from vad.energy_vad import EnergyVAD
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import concurrent.futures
import threading
from queue import Queue, Empty

# Constants
MAX_SEGMENT_LENGTH = 30000  # 30 seconds in milliseconds
MIN_SEGMENT_LENGTH = 1000   # 1 second in milliseconds
MAX_SEGMENT_LENGTH_CHARS = 9999 
WITH_TIMESTAMPS = False # Set to False, True (the worst option), or "word" to get timestamps for each word
MIN_GPU_MEMORY = 23000  # 23GB in MB
BATCH_SIZE_PER_GPU = 16  # Number of segments to process in a single batch on each GPU

# Parent directory for all temporary files
TEMP_PARENT_DIR = "temp"

# Thread-local storage for model instances
thread_local = threading.local()

# Dictionary to store models by GPU ID - initialized once and reused
gpu_models = {}
gpu_models_lock = threading.Lock()

# Get available GPUs once at module import
available_gpus = []
default_device = None

def initialize_models():
    """
    Initialize models on all available GPUs.
    Call this once at program startup before processing any audio files.
    """
    global available_gpus, default_device
    
    # Find available GPUs
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        default_device = torch.device('cpu')
        print("No suitable GPU available, using CPU instead.")
        return
    
    # Initialize models on all GPUs in advance
    model_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(model_name)
    
    print(f"Initializing models on {len(available_gpus)} GPUs...")
    for gpu_id in available_gpus:
        print(f"Loading model on GPU {gpu_id}")
        device = torch.device(f'cuda:{gpu_id}')
        model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager").to(device)
        model.generation_config.language = "he"
        
        asr = pipeline("automatic-speech-recognition", 
                        model=model, 
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor, 
                        device=device)
        
        gpu_models[gpu_id] = {
            'model': model,
            'processor': processor,
            'asr': asr,
            'device': device
        }
    
    default_device = torch.device(f'cuda:{available_gpus[0]}')
    print(f"Models initialized on {len(available_gpus)} GPUs: {available_gpus}")

def get_available_gpus():
    """
    Get available GPUs with more than MIN_GPU_MEMORY free memory
    Returns a list of GPU indices
    """
    if not torch.cuda.is_available():
        print("No GPU available, using CPU instead.")
        return []
        
    import subprocess
    mem_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv']).decode('utf-8').split('\n')
    mem_output = mem_output[1:-1]  # Skip header and empty last line
    
    available_gpus = []
    for i, mem_str in enumerate(mem_output):
        try:
            mem = int(mem_str.replace(' MiB', ''))
            if mem > MIN_GPU_MEMORY:
                available_gpus.append(i)
                print(f"GPU {i} available with {mem} MiB free memory")
        except:
            continue
            
    return available_gpus

# Load the model processor once
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)

# Get model instance for the current thread/GPU
def get_model_instance(gpu_id):
    """
    Get a model instance for the current thread.
    First checks in the global gpu_models dict, then falls back to thread_local storage.
    
    Args:
        gpu_id: The GPU ID to use
        
    Returns:
        (asr, device) tuple
    """
    # First check if we have a pre-initialized model
    if gpu_id in gpu_models:
        return gpu_models[gpu_id]['asr'], gpu_models[gpu_id]['device']
    
    # If not, check if this thread already has a model
    if not hasattr(thread_local, 'model') or thread_local.gpu_id != gpu_id:
        print(f"Loading model on GPU {gpu_id} (thread local)")
        device = torch.device(f'cuda:{gpu_id}')
        thread_local.model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager").to(device)
        thread_local.model.generation_config.language = "he"
        thread_local.device = device
        thread_local.asr = pipeline("automatic-speech-recognition", 
                              model=thread_local.model, 
                              tokenizer=processor.tokenizer,
                              feature_extractor=processor.feature_extractor, 
                              device=device)
        thread_local.gpu_id = gpu_id
    
    return thread_local.asr, thread_local.device

# Function to split audio file into segments
def normalize_audio(audio):
    """
    Normalize audio to have zero mean
    """
    audio_mean = audio.mean()
    audio_std = audio.std()
    print(audio_mean, "\t", audio_std)
    if audio_std > 0:
        audio = (audio - audio_mean) #/ audio_std
    return audio

def split_audio(audio_file, output_dir):
    """
    Args:
        audio_file: Path to the input audio file.
        output_dir: Directory to save the temporary audio segments.

    Returns:
        A list of file names for the created audio segments.
    """
    ENERGY_THRESHOLD = 0.005
    FRAME_LENGTH = 20  # in milliseconds
    vad = EnergyVAD(
        sample_rate=16000,
        frame_length=FRAME_LENGTH,
        frame_shift=FRAME_LENGTH,
        energy_threshold=ENERGY_THRESHOLD,
        pre_emphasis=0.95,
    )

    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Normalize audio before VAD
    normalized_audio = normalize_audio(audio)
    voice_activity = vad(normalized_audio)

    # Apply median filter to smooth the voice activity detection
    voice_activity_median = scipy.signal.medfilt(voice_activity, kernel_size=7)
    segments = []
    segment_files = []
    segment_lengths = []
    forced_splits = []
    start = 0
    total_frames = len(voice_activity_median)

    base_filename = os.path.splitext(os.path.basename(audio_file))[0]

    while start < total_frames:
        max_frames = min(start + MAX_SEGMENT_LENGTH // FRAME_LENGTH - 1, total_frames)
        min_frames = start + MIN_SEGMENT_LENGTH // FRAME_LENGTH
        found_silence = False
        
        for end in range(max_frames, min_frames, -1):
            if end >= len(voice_activity_median):
                end = len(voice_activity_median) - 1
            if not voice_activity_median[end]:
                found_silence = True
                break
        
        if not found_silence:
            end = max_frames
            forced_splits.append(len(segments))
            print(f"Warning: Forced to split at maximum length near {(start * FRAME_LENGTH) / 1000:.2f} seconds")

        # Use original audio for saving segments
        segment = audio[start * FRAME_LENGTH * sr // 1000:end * FRAME_LENGTH * sr // 1000]
        segment_length = len(segment) / sr
        segment_lengths.append(segment_length)
        segments.append(segment)

        segment_filename = f"{base_filename}_{len(segments):03d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        segment_files.append(segment_filename)

        start = end + 1

    natural_segments_lengths = [length for idx, length in enumerate(segment_lengths) if idx not in forced_splits]
    forced_segments_lengths = [length for idx, length in enumerate(segment_lengths) if idx in forced_splits]
    
    if natural_segments_lengths:
        sorted_lengths = sorted(natural_segments_lengths)
        print("\nSegment Statistics:")
        print("5 Shortest segments (seconds):", [f"{x:.2f}" for x in sorted_lengths[:min(5, len(sorted_lengths))]])
        print("5 Longest natural segments (seconds):", [f"{x:.2f}" for x in sorted_lengths[-min(5, len(sorted_lengths)):]])
        print(f"Median length: {sorted_lengths[len(sorted_lengths)//2]:.2f} seconds")
        print(f"Average length: {sum(natural_segments_lengths)/len(natural_segments_lengths):.2f} seconds")
    
    print(f"Total segments: {len(segment_lengths)}")
    print(f"Number of forced splits: {len(forced_splits)}")
    if forced_splits:
        print(f"Forced split segments lengths: {[f'{forced_segments_lengths[i]:.2f}' for i in range(len(forced_splits))]}\n")
    
    if len(segments) > 1:
        last_segment_duration = len(segments[-1]) / sr
        if last_segment_duration < 0.1:
            print(f"Removing last segment (too short: {last_segment_duration:.2f} seconds)")
            os.remove(os.path.join(output_dir, segment_files[-1]))
            segment_files.pop()
            segments.pop()
    
    return segment_files

def create_srt_segment(result, start_time=0, last_index=0, segment_duration=None):
    """
    Creates a list of Subtitle objects from the Whisper result.

    Args:
        result: The output from the Whisper pipeline.
        start_time: The starting time (in seconds) for this segment within the full audio.
        last_index: The last subtitle index from the previous segment.
        segment_duration: The duration of the audio segment in seconds.

    Returns:
        A list of Subtitle objects and the last index used.
    """
    subtitles = []
    current_line = ""
    segment_start_time = None
    current_index = last_index + 1

    for i, segment in enumerate(result["chunks"]):
        word = segment["text"].strip()
        if segment_start_time is None:
            segment_start_time = segment["timestamp"][0]

        if len(current_line + word) > MAX_SEGMENT_LENGTH_CHARS:
            # Improved handling of timestamp values
            end_time = segment["timestamp"][0] + start_time
            subtitles.append(srt.Subtitle(index=current_index,
                                          start=timedelta(seconds=segment_start_time + start_time),
                                          end=timedelta(seconds=end_time),
                                          content=current_line.strip()))
            current_index += 1
            current_line = word + " "
            segment_start_time = segment["timestamp"][0]
        else:
            current_line += word + " "

    # Add the last subtitle if there's remaining content
    if current_line:
        last_timestamp = result["chunks"][-1]["timestamp"]
        # Improved handling of timestamp values - handle None values
        end_time_value = last_timestamp[1] if last_timestamp[1] is not None else last_timestamp[0]
        
        # Ensure the last subtitle ends at least at the end of the segment
        if segment_duration is not None and i == len(result["chunks"]) - 1:
            segment_end_time = start_time + segment_duration
            if end_time_value + start_time < segment_end_time:
                end_time_value = segment_duration  # Use segment_duration directly as we'll add start_time later
        
        subtitles.append(srt.Subtitle(index=current_index,
                                      start=timedelta(seconds=segment_start_time + start_time),
                                      end=timedelta(seconds=end_time_value + start_time),
                                      content=current_line.strip()))
        current_index += 1

    return subtitles, current_index - 1

def process_audio_segment(audio_segment, start_time=0, last_index=0):
    """
    Processes a single audio segment to generate the SRT content.

    Args:
        audio_segment: The audio data for this segment.
        start_time: The starting time (in seconds) for this segment within the full audio.
        last_index: The last subtitle index from the previous segment.

    Returns:
        A list of Subtitle objects for the given segment and the last index used.
    """
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=device)

    result = asr(audio_segment, return_timestamps=WITH_TIMESTAMPS,
                 generate_kwargs={"language": "<|he|>",
                                  "task": "transcribe"})
    
    segment_duration = len(audio_segment) / 16000  # Calculate duration in seconds
    
    if not WITH_TIMESTAMPS:
        result["chunks"] = [{'text': result["text"], 'timestamp': (0, segment_duration)}]
    
    subtitles, last_index = create_srt_segment(result, start_time, last_index, segment_duration)
    return subtitles, last_index

def process_audio_segments_batch(audio_segments, start_times, last_index):
    """
    Processes multiple audio segments in a batch to generate SRT content.

    Args:
        audio_segments: List of audio segments data
        start_times: List of start times for each segment
        last_index: The last subtitle index from the previous batch

    Returns:
        A list of Subtitle objects and the last index used
    """
    # Determine optimal batch size based on available GPUs
    effective_batch_size = len(audio_segments)
    # if len(available_gpus) > 1:
    #     # When using DataParallel, we can process larger batches efficiently
    #     print(f'Processing batch of {effective_batch_size} segments across {len(available_gpus)} GPUs')
    # else:
    #     print(f'Processing batch of {effective_batch_size} segments')
    
    asr = pipeline("automatic-speech-recognition", 
                  model=model, 
                  tokenizer=processor.tokenizer,
                  feature_extractor=processor.feature_extractor, 
                  device=device)

    results = asr(audio_segments, return_timestamps=WITH_TIMESTAMPS,
                 batch_size=effective_batch_size,
                 generate_kwargs={"language": "<|he|>",
                               "task": "transcribe"})
    
    all_subtitles = []
    current_index = last_index

    # Process each result in the batch
    for idx, (result, start_time, audio_segment) in enumerate(zip(results, start_times, audio_segments)):
        segment_duration = len(audio_segment) / 16000  # Calculate duration in seconds
        
        if not WITH_TIMESTAMPS:
            # Calculate duration based on the current segment length
            result = {"chunks": [{'text': result["text"], 
                                'timestamp': (0, segment_duration)}]}
        
        subtitles, current_index = create_srt_segment(result, start_time, current_index, segment_duration)
        all_subtitles.extend(subtitles)

    return all_subtitles, current_index

def process_segment_on_gpu(args):
    """Process a single segment on a specific GPU"""
    segment, start_time, last_index, gpu_id = args
    
    # Get model instance for this thread/GPU
    asr, device = get_model_instance(gpu_id)
    
    result = asr(segment, return_timestamps=WITH_TIMESTAMPS,
                 generate_kwargs={"language": "<|he|>",
                                  "task": "transcribe"})
    
    segment_duration = len(segment) / 16000  # Calculate duration in seconds
    
    if not WITH_TIMESTAMPS:
        result = {"chunks": [{'text': result["text"], 
                            'timestamp': (0, segment_duration)}]}
    
    subtitles, last_index = create_srt_segment(result, start_time, last_index, segment_duration)
    return subtitles, last_index

def process_batch_on_gpu(batch_items, gpu_id):
    """
    Process a batch of segments on a specific GPU
    
    Args:
        batch_items: List of (file_id, idx, segment, start_time) tuples
        gpu_id: GPU ID to use for processing
        
    Returns:
        Dictionary mapping (file_id, segment_idx) to subtitle list
    """
    # Get model instance for this thread/GPU
    asr, device = get_model_instance(gpu_id)
    
    # Extract data from batch items
    file_ids = [item[0] for item in batch_items]
    indices = [item[1] for item in batch_items]
    segments = [item[2] for item in batch_items]
    start_times = [item[3] for item in batch_items]
    
    batch_size = len(segments)
    segment_ids = [f"{file_id}:{idx+1}" for file_id, idx in zip(file_ids, indices)]
    
    # print(f"GPU {gpu_id} processing batch of {batch_size} segments: {segment_ids}")
    
    # Process batch all at once
    results = asr(segments, return_timestamps=WITH_TIMESTAMPS,
                 batch_size=batch_size,  # Process all segments in batch
                 generate_kwargs={"language": "<|he|>",
                                 "task": "transcribe"})
    
    # Process results and create subtitles
    batch_results = {}
    
    for i, (result, start_time, segment) in enumerate(zip(results, start_times, segments)):
        segment_duration = len(segment) / 16000  # Calculate duration in seconds
        
        if not WITH_TIMESTAMPS:
            # Format result for create_srt_segment
            result = {"chunks": [{'text': result["text"], 
                                'timestamp': (0, segment_duration)}]}
        
        # Use segment index as the last_index to ensure unique numbering
        subtitles, _ = create_srt_segment(result, start_time, indices[i], segment_duration)
        batch_results[(file_ids[i], indices[i])] = subtitles
        
    # print(f"GPU {gpu_id} completed batch of {batch_size} segments: {segment_ids}")
    return batch_results

def generate_multiple_srt_from_audio(audio_files, output_srt_files):
    """
    Generates multiple SRT files from audio files, processing segments across all files
    optimally on multiple GPUs.

    Args:
        audio_files: List of paths to the input audio files.
        output_srt_files: List of paths to the output SRT files.
    """
    if len(audio_files) != len(output_srt_files):
        raise ValueError("Number of audio files must match number of output SRT files")
    
    # Ensure the parent temp directory exists
    os.makedirs(TEMP_PARENT_DIR, exist_ok=True)
        
    # Create unique output directories for segments inside the parent temp directory
    temp_dirs = []
    for audio_file in audio_files:
        file_id = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(TEMP_PARENT_DIR, f"audio_segments_{file_id}")
        os.makedirs(output_dir, exist_ok=True)
        temp_dirs.append(output_dir)

    print(f"Processing {len(audio_files)} audio files with cross-file batching")
    if available_gpus:
        print(f"Using {len(available_gpus)} GPUs: {available_gpus}")
        print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    else:
        print("Using CPU")
    
    # Process all files at once - first split them into segments
    all_segments_info = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(audio_files))) as executor:
        # Split all files into segments in parallel
        split_futures = []
        
        for i, (audio_file, temp_dir) in enumerate(zip(audio_files, temp_dirs)):
            print(f"Splitting audio file ({i+1}/{len(audio_files)}): {audio_file}")
            future = executor.submit(split_audio, audio_file, temp_dir)
            split_futures.append((future, i, audio_file, temp_dir))
        
        # Collect segment information
        for future, i, audio_file, temp_dir in split_futures:
            try:
                segment_files = future.result()
                file_id = os.path.splitext(os.path.basename(audio_file))[0]
                
                print(f"File {i+1}/{len(audio_files)}: {file_id} - {len(segment_files)} segments")
                
                # Load all segments for this file
                file_segments = []
                file_start_times = []
                file_segment_paths = []
                total_duration = 0
                
                for j, segment_file in enumerate(segment_files):
                    segment_path = os.path.join(temp_dir, segment_file)
                    file_segment_paths.append(segment_path)
                    audio_segment, sr = librosa.load(segment_path, sr=16000)
                    file_segments.append(audio_segment)
                    file_start_times.append(total_duration)
                    segment_duration = len(audio_segment) / sr
                    total_duration += segment_duration
                
                # Store all segment info for this file
                all_segments_info.append({
                    'file_id': file_id,
                    'file_index': i,
                    'segments': file_segments,
                    'start_times': file_start_times,
                    'segment_paths': file_segment_paths,
                    'segment_count': len(segment_files)
                })
                
            except Exception as e:
                print(f"Error splitting file {audio_file}: {e}")
    
    # Now process all segments across all files on GPUs
    if not all_segments_info:
        print("No segments found to process")
        return
    
    # Flatten all segments across files into a single processing queue
    all_segments = []
    all_file_ids = []
    all_segment_indices = []
    all_start_times = []
    
    for file_info in all_segments_info:
        for j, (segment, start_time) in enumerate(zip(file_info['segments'], file_info['start_times'])):
            all_segments.append(segment)
            all_file_ids.append(file_info['file_id'])
            all_segment_indices.append(j)
            all_start_times.append(start_time)
    
    print(f"Total segments across all files: {len(all_segments)}")
    
    # Process all segments on GPU in optimal batches
    segment_results = {}
    
    if available_gpus:
        # Create a task queue for all segments
        task_queue = Queue()
        lock = threading.Lock()
        
        # Fill the queue with all segment indices
        for i in range(len(all_segments)):
            task_queue.put(i)
            
        # Create a progress bar for batch processing
        pbar = tqdm(total=len(all_segments), desc="Processing segments")
        
        def worker(gpu_id):
            """Worker function that processes batches from the queue"""
            while True:
                batch_items = []
                
                # Try to fill a batch
                try:
                    for _ in range(BATCH_SIZE_PER_GPU):
                        try:
                            # Get next segment index from queue with timeout
                            idx = task_queue.get(block=False)
                            batch_items.append((
                                all_file_ids[idx], 
                                all_segment_indices[idx], 
                                all_segments[idx], 
                                all_start_times[idx]
                            ))
                        except Empty:
                            break
                except Exception as e:
                    print(f"Error building batch: {e}")
                    break
                
                # If batch is empty, exit the worker
                if not batch_items:
                    break
                    
                # Process the batch
                try:
                    batch_results = process_batch_on_gpu(batch_items, gpu_id)
                    
                    # Store results and mark tasks as done
                    with lock:
                        for key, subtitles in batch_results.items():
                            segment_results[key] = subtitles
                        # Update progress bar
                        pbar.update(len(batch_items))
                        
                    # Mark all tasks in this batch as done
                    for _ in range(len(batch_items)):
                        task_queue.task_done()
                        
                except Exception as e:
                    print(f"Error processing batch on GPU {gpu_id}: {e}")
                    # Put segments back in the queue
                    for _ in range(len(batch_items)):
                        task_queue.task_done()  # Mark original as done
                    
                    # Only retry if it's not a GPU error
                    if "CUDA" not in str(e) and "GPU" not in str(e):
                        print("Requeuing segments due to non-GPU error")
                        for i in range(len(batch_items)):
                            task_queue.put(i)  # Put back in queue
        
        # Create worker threads for each GPU
        threads = []
        for gpu_id in available_gpus:
            thread = threading.Thread(target=worker, args=(gpu_id,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Wait for all threads to complete
        task_queue.join()
        pbar.close()
    
    else:
        # Fall back to sequential processing on CPU
        print("Using CPU for processing - this will be slow")
        # Implementation for CPU if needed
    
    # Group results by file and save each SRT
    for file_info in all_segments_info:
        file_id = file_info['file_id']
        file_index = file_info['file_index']
        output_file = output_srt_files[file_index]
        
        # Collect all subtitles for this file
        file_subtitles = []
        for i in range(file_info['segment_count']):
            key = (file_id, i)
            if key in segment_results:
                file_subtitles.extend(segment_results[key])
        
        # Sort subtitles by start time
        file_subtitles.sort(key=lambda s: s.start)
        
        # Renumber subtitles
        for i, subtitle in enumerate(file_subtitles):
            subtitle.index = i + 1
        
        # Write SRT file
        full_srt_content = srt.compose(file_subtitles)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_srt_content)
            
        print(f"Completed SRT file: {output_file}")
    
    # Clean up temporary files
    for file_info in all_segments_info:
        temp_dir = os.path.dirname(file_info['segment_paths'][0])
        for segment_path in file_info['segment_paths']:
            try:
                os.remove(segment_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

def generate_srt_from_audio(audio_file, output_srt_file):
    """
    Generates an SRT file from an audio file.
    Wrapper function that calls generate_multiple_srt_from_audio.

    Args:
        audio_file: Path to the input audio file.
        output_srt_file: Path to the output SRT file.
    """
    generate_multiple_srt_from_audio([audio_file], [output_srt_file])

def main():
    # Ensure the parent temp directory exists
    os.makedirs(TEMP_PARENT_DIR, exist_ok=True)
    
    audio_file = "test.wav"
    output_srt_file = "output.srt"
    generate_srt_from_audio(audio_file, output_srt_file)

if __name__ == "__main__":
    initialize_models()
    main()