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

# Constants
MAX_SEGMENT_LENGTH = 30000  # 30 seconds in milliseconds
MIN_SEGMENT_LENGTH = 1000   # 1 second in milliseconds
MAX_SEGMENT_LENGTH_CHARS = 9999 
WITH_TIMESTAMPS = False # Set to False, True (the worst option), or "word" to get timestamps for each word

if torch.cuda.is_available():
    # choose the GPU with empty memory
    import subprocess
    mem = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv']).decode('utf-8').split('\n')
    mem = mem[1:-1]  # Skip header and empty last line
    mem = [int(m.replace(' MiB', '')) for m in mem]
    device = torch.device(f'cuda:{mem.index(max(mem))}' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU instead.")

# Load the model and processor
model_name = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager").to(device)
processor = WhisperProcessor.from_pretrained(model_name)
model.generation_config.language = "he"

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
    ENERGY_THRESHOLD = 0.002
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
    voice_activity_median = scipy.signal.medfilt(voice_activity, kernel_size=15)
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
    asr = pipeline("automatic-speech-recognition", 
                  model=model, 
                  tokenizer=processor.tokenizer,
                  feature_extractor=processor.feature_extractor, 
                  device=device)

    print(f'Processing batch of {len(audio_segments)} segments')
    
    results = asr(audio_segments, return_timestamps=WITH_TIMESTAMPS,
                 batch_size=len(audio_segments),
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

def generate_srt_from_audio(audio_file, output_srt_file, batch_size=4):
    """
    Generates an SRT file from an audio file, processing segments in batches.

    Args:
        audio_file: Path to the input audio file.
        output_srt_file: Path to the output SRT file.
        batch_size: Number of segments to process simultaneously
    """
    output_dir = "temp_audio_segments"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing audio file: {audio_file}")
    print(f"Output SRT file: {output_srt_file}")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")

    segment_files = split_audio(audio_file, output_dir)
    
    all_subtitles = []
    total_duration = 0
    last_index = 0
    
    total_segments = len(segment_files)
    num_batches = (total_segments + batch_size - 1) // batch_size
    print(f"\nProcessing {total_segments} segments in {num_batches} batches...")
    
    # Process segments in batches
    for i in tqdm(range(0, len(segment_files), batch_size)):
        batch_files = segment_files[i:i + batch_size]
        batch_segments = []
        batch_start_times = []
        
        print(f"\nProcessing batch {(i // batch_size) + 1}/{num_batches} "
              f"(segments {i + 1}-{min(i + batch_size, total_segments)} of {total_segments})")
        
        # Load audio segments for current batch
        for segment_file in batch_files:
            segment_path = os.path.join(output_dir, segment_file)
            audio_segment, _ = librosa.load(segment_path, sr=16000)
            batch_segments.append(audio_segment)
            batch_start_times.append(total_duration)
            total_duration += librosa.get_duration(y=audio_segment, sr=16000)
        
        # Process the batch
        batch_subtitles, last_index = process_audio_segments_batch(
            batch_segments, 
            batch_start_times, 
            last_index
        )
        all_subtitles.extend(batch_subtitles)
    
    # Compose the final SRT content from all subtitles
    full_srt_content = srt.compose(all_subtitles)
    
    with open(output_srt_file, "w", encoding="utf-8") as f:
        f.write(full_srt_content)

    # Cleanup
    for segment_file in segment_files:
        os.remove(os.path.join(output_dir, segment_file))
    os.rmdir(output_dir)
    
    print(f"\nCompleted! SRT file saved to: {output_srt_file}")

def main():
    audio_file = "test.wav"
    output_srt_file = "output.srt"
    generate_srt_from_audio(audio_file, output_srt_file)

if __name__ == "__main__":
    main()