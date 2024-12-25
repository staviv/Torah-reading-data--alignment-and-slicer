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




# Define the directory containing the audio files
audio_dir = "/home/prj8045/data/Nusach-Sephardic-Yerushalmi-Avi-Zarki"
audio_dir = "/home/prj8045/data/abc"


# Process all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(('.mp3', '.wav')):  # Add more audio extensions if needed
        audio_path = os.path.join(audio_dir, filename)
        srt_path = os.path.splitext(audio_path)[0] + '.srt'
        
        print(f"Processing: {filename}")
        generate_srt_from_audio(audio_path, srt_path, batch_size=32)
