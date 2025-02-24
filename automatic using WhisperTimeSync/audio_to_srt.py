import os
import librosa
import soundfile as sf
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
MAX_SEGMENT_LENGTH_CHARS = 8000 
WITH_TIMESTAMPS = False

# choose the GPU with empty memory
import subprocess
mem = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv']).decode('utf-8').split('\n')
mem = mem[1:-1]  # Skip header and empty last line
mem = [int(m.replace(' MiB', '')) for m in mem]
device = torch.device(f'cuda:{mem.index(max(mem))}' if torch.cuda.is_available() else 'cpu')


# Load the model and processor
model_name = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)
model.generation_config.language = "he"



import numpy as np

def normalize_audio(audio, target_amplitude=0.99):
    """Normalizes the audio signal to a target peak amplitude.

    Args:
        audio: A numpy array representing the audio signal.
        target_amplitude: The desired peak amplitude (between 0 and 1).

    Returns:
        A numpy array representing the normalized audio signal.
    """
    peak_amplitude = np.max(np.abs(audio))
    if peak_amplitude == 0:
        return audio  # Avoid division by zero for silent audio
    print(f"Peak amplitude: {peak_amplitude}")
    scaling_factor = target_amplitude / peak_amplitude
    normalized_audio = audio * scaling_factor
    return normalized_audio


# Function to split audio file into segments
def split_audio(audio_file, output_dir):
    """
    Args:
        audio_file: Path to the input audio file.
        output_dir: Directory to save the temporary audio segments.

    Returns:
        A list of file names for the created audio segments.
    """
    FRAME_LENGTH = 20  # in milliseconds
    vad = EnergyVAD(
        sample_rate=16000,
        frame_length=FRAME_LENGTH,
        frame_shift=FRAME_LENGTH,
        energy_threshold=0.002,
        pre_emphasis=0.95,
    )

    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Normalize audio before VAD
    normalized_audio = normalize_audio(audio)
    voice_activity = vad(normalized_audio)
    # Apply median filter to smooth the voice activity detection
    voice_activity_median = scipy.signal.medfilt(voice_activity, kernel_size=9)
    print("the indices of zeros in voice_activity_median are: ", np.where(voice_activity_median == 0)[0])
    segments = []
    segment_files = []
    start = 0
    total_frames = len(voice_activity_median)

    base_filename = os.path.splitext(os.path.basename(audio_file))[0]

    while start < total_frames:
        for end in range(min(start + MAX_SEGMENT_LENGTH // FRAME_LENGTH - 1, total_frames), start, -1):
            if end >= len(voice_activity_median):
                end = len(voice_activity_median) - 1
            if not voice_activity_median[end]:
                break

        # Use original audio for saving segments
        segment = audio[start * FRAME_LENGTH * sr // 1000:end * FRAME_LENGTH * sr // 1000]
        segments.append(segment)

        segment_filename = f"{base_filename}_{len(segments):03d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        segment_files.append(segment_filename)

        start = end + 1

    # if the last one is too short (less than 0.1 seconds), we throw it away
    if len(segments) > 1:
        last_segment_duration = len(segments[-1]) / sr
        if last_segment_duration < 0.1:
            os.remove(os.path.join(output_dir, segment_files[-1]))
            segment_files.pop()
            segments.pop()
    
    return segment_files
    



def create_srt_segment(result, start_time=0, last_index=0):
    """
    Creates a list of Subtitle objects from the Whisper result.

    Args:
        result: The output from the Whisper pipeline.
        start_time: The starting time (in seconds) for this segment within the full audio.
        last_index: The last subtitle index from the previous segment.

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
        subtitles.append(srt.Subtitle(index=current_index,
                                      start=timedelta(seconds=segment_start_time + start_time),
                                      end=timedelta(seconds=result["chunks"][-1]["timestamp"][1] + start_time),
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
    if not WITH_TIMESTAMPS:
        result["chunks"] = [{'text': result["text"], 'timestamp': (0, (len(audio_segment) // 160)/100)}] # timestamp is in seconds with 2 decimal places
    subtitles, last_index = create_srt_segment(result, start_time, last_index)
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

    results = asr(audio_segments, return_timestamps=WITH_TIMESTAMPS,
                 batch_size=len(audio_segments),
                 generate_kwargs={"language": "<|he|>",
                               "task": "transcribe"})
    
    all_subtitles = []
    current_index = last_index

    # Process each result in the batch
    for idx, (result, start_time, audio_segment) in enumerate(zip(results, start_times, audio_segments)):
        if not WITH_TIMESTAMPS:
            # Calculate duration based on the current segment length
            duration = (len(audio_segment) // 160) / 100
            result = {"chunks": [{'text': result["text"], 
                                'timestamp': (0, duration)}]}
        
        subtitles, current_index = create_srt_segment(result, start_time, current_index)
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

    segment_files = split_audio(audio_file, output_dir)
    
    all_subtitles = []
    total_duration = 0
    last_index = 0
    
    # Process segments in batches
    for i in tqdm(range(0, len(segment_files), batch_size)):
        batch_files = segment_files[i:i + batch_size]
        batch_segments = []
        batch_start_times = []
        
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