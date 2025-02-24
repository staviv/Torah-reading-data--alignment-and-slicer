import json
from get_torah_text_using_sefaria import get_chapter_string
import scipy.io.wavfile as wavfile
import io
import pathlib
import soundfile as sf
from tqdm import tqdm
from nikud_and_teamim import remove_nikud, replace_teamim_with_emphasis, remove_nikud_and_teamim, remove_nikud_dicta

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import srt
from datetime import timedelta
from transformers import pipeline
import os
import soundfile as sf
import scipy.signal
from vad import EnergyVAD 
from tqdm import tqdm

from audio_to_srt import generate_srt_from_audio, device
from create_steps_files import create_steps_files, remove_steps_files
import subprocess



# Define the directory containing the audio files
# audio_dir = "/home/prj8045/data/Nusach-Sephardic-Yerushalmi-Avi-Zarki/"
audio_dir = "/home/prj8045/data/abc/"

whisperTimeSync = "/home/prj8045/Torah-reading-data--alignment-and-slicer/automatic using WhisperTimeSync/WhisperTimeSync/distrib/WhisperTimeSync.jar"
def iterative_sync_text_srt(text_path, whisperTimeSync):
    """
    Iteratively synchronize the text and the low-quality SRT file.
    """
    # Load the text and the low-quality SRT file
    

    create_steps_files(text_path)
    subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'output.srt', 'step01.txt', 'he'], check=True)
    subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step01.txt.srt', 'step02.txt', 'he'], check=True)
    subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step02.txt.srt', 'step03.txt', 'he'], check=True)
    subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step03.txt.srt', 'step04.txt', 'he'], check=True)
    subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step04.txt.srt', 'final_step.txt', 'he'], check=True)
    


# Process all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(('.mp3', '.wav')):  # Add more audio extensions if needed
        audio_path = os.path.join(audio_dir, filename)
        # Create path for raw SRT
        raw_srt_path = os.path.splitext(audio_path)[0] + '_RAW.srt'
        # Create path for text file (same directory as audio, but .txt extension)
        text_path = os.path.splitext(audio_path)[0] + '.txt'
        # Create path for final SRT
        final_srt_path = os.path.splitext(audio_path)[0] + '.srt'
        
        print(f"Processing: {filename}")
        # Generate initial SRT
        generate_srt_from_audio(audio_path, raw_srt_path, batch_size=4)
        
        # Sync with text file
        iterative_sync_text_srt(text_path, whisperTimeSync)
        
        # Move final synced SRT to destination
        os.rename("final_step.txt.srt", final_srt_path)
        
        # Clean up temporary files
        remove_steps_files()



