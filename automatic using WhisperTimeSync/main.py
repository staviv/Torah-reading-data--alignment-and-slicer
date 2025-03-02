import os
from audio_to_srt import generate_srt_from_audio
from create_steps_files import create_steps_files, remove_steps_files
import subprocess



# Define the directory containing the audio files
# audio_dir = "/home/prj8045/data/Nusach-Sephardic-Yerushalmi-Avi-Zarki/"
audio_dir = "/home/prj8045/data/abc/"
text_dir = "/home/prj8045/data/text/"

whisperTimeSync = "/home/prj8045/Torah-reading-data--alignment-and-slicer/automatic using WhisperTimeSync/WhisperTimeSync/distrib/WhisperTimeSync.jar"

def one_step_sync_text_srt(low_quality_srt_path, text_path, whisperTimeSync):
    """
    Synchronize the text and the low-quality SRT file.
    """
    cwd = os.getcwd() + "/temp"
    subprocess.run(
        ['java', '-Xmx2G', '-jar', whisperTimeSync, low_quality_srt_path, text_path, 'he'],
        check=True,
        cwd=cwd
    )

def iterative_sync_text_srt(text_path, raw_srt_path, whisperTimeSync):
    """
    Iteratively synchronize the text and the low-quality SRT file.
    """
    # Load the text and the low-quality SRT file
    # create temp directory
    os.makedirs("temp", exist_ok=True)
    # copy srt file to temp
    os.system(f"cp {raw_srt_path} temp/")
    
    create_steps_files(text_path)
    one_step_sync_text_srt(os.path.basename(raw_srt_path), "step01.txt", whisperTimeSync)
    one_step_sync_text_srt("step01.txt.srt", "step02.txt", whisperTimeSync)
    one_step_sync_text_srt("step02.txt.srt", "step03.txt", whisperTimeSync)
    one_step_sync_text_srt("step03.txt.srt", "step04.txt", whisperTimeSync)
    one_step_sync_text_srt("step04.txt.srt", "final_step.txt", whisperTimeSync)

    # subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, raw_srt_path, 'step01.txt', 'he'], check=True)
    # subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step01.txt.srt', 'step02.txt', 'he'], check=True)
    # subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step02.txt.srt', 'step03.txt', 'he'], check=True)
    # subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step03.txt.srt', 'step04.txt', 'he'], check=True)
    # subprocess.run(['java', '-Xmx2G', '-jar', whisperTimeSync, 'step04.txt.srt', 'final_step.txt', 'he'], check=True)
    


# Process all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(('.mp3', '.wav')):  # Add more audio extensions if needed
        audio_path = os.path.join(audio_dir, filename)
        # Create path for text file (same directory as audio, but .txt extension)
        text_path = os.path.join(text_dir, os.path.splitext(filename)[0] + '.txt')        
        # Create path for raw SRT
        raw_srt_path = os.path.splitext(audio_path)[0] + '_RAW.srt'
        # Create path for final SRT
        final_srt_path = os.path.splitext(audio_path)[0] + '.srt'
        
        print(f"Processing: {filename}")
        
        # Generate initial SRT
        generate_srt_from_audio(audio_path, raw_srt_path, batch_size=1)
        
        # Sync with text file
        iterative_sync_text_srt(text_path, raw_srt_path, whisperTimeSync)
        
        # Move final synced SRT to destination
        os.rename(os.getcwd() + "/temp/final_step.txt.srt", final_srt_path)
        
        # Clean up temporary files
        remove_steps_files()



