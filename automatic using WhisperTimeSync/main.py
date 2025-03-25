import os
import threading
import concurrent.futures
from audio_to_srt import multiple_generate_srt_from_audio, initialize_models
from create_steps_files import create_steps_files, remove_steps_files
import subprocess

# Define the directory containing the audio files
audio_dir = "/home/prj8045/train_data/Maroco-Michael-Bitton/"
text_dir = "/home/prj8045/train_data/text/"

whisperTimeSync = "/home/prj8045/Torah-reading-data--alignment-and-slicer/automatic using WhisperTimeSync/WhisperTimeSync/distrib/WhisperTimeSync.jar"

# Lock for synchronizing temp directory creation and cleanup
temp_dir_lock = threading.Lock()

# Create a parent temp directory
TEMP_PARENT_DIR = "temp"

def one_step_sync_text_srt(low_quality_srt_path, text_path, whisperTimeSync, temp_dir):
    """
    Synchronize the text and the low-quality SRT file.
    """
    subprocess.run(
        ['java', '-Xmx2G', '-jar', whisperTimeSync, low_quality_srt_path, text_path, 'he'],
        check=True,
        cwd=temp_dir
    )

def iterative_sync_text_srt(text_path, raw_srt_path, whisperTimeSync, file_id):
    """
    Iteratively synchronize the text and the low-quality SRT file.
    Uses a unique temp directory for each file to avoid conflicts.
    """
    # Create a temp parent directory if it doesn't exist
    with temp_dir_lock:
        os.makedirs(TEMP_PARENT_DIR, exist_ok=True)
    
    # Create a unique temp directory for this file inside the parent temp directory
    temp_dir = os.path.join(TEMP_PARENT_DIR, f"temp_{file_id}")
    with temp_dir_lock:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Copy SRT file to temp directory
    subprocess.run(["cp", raw_srt_path, f"{temp_dir}/"])
    
    # Create steps files in the temp directory
    create_steps_files(text_path, temp_dir)
    
    # Run the synchronization steps
    one_step_sync_text_srt(os.path.basename(raw_srt_path), "step01.txt", whisperTimeSync, temp_dir)
    one_step_sync_text_srt("step01.txt.srt", "step02.txt", whisperTimeSync, temp_dir)
    one_step_sync_text_srt("step02.txt.srt", "step03.txt", whisperTimeSync, temp_dir)
    one_step_sync_text_srt("step03.txt.srt", "step04.txt", whisperTimeSync, temp_dir)
    one_step_sync_text_srt("step04.txt.srt", "final_step.txt", whisperTimeSync, temp_dir)
    
    final_path = os.path.join(temp_dir, "final_step.txt.srt")
    return final_path, temp_dir

def main():
    """
    Main function to process all audio files at once for optimal GPU utilization
    """
    # Initialize models once before processing any files
    initialize_models()
    
    # Create the temp parent directory
    os.makedirs(TEMP_PARENT_DIR, exist_ok=True)
    
    # Get list of all audio files to be processed
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]
    total_files = len(audio_files)
    
    print(f"Found {total_files} audio files to process")
    print(f"Processing all files together for optimal GPU batch utilization")
    
    # # Create paths for all files
    audio_paths = [os.path.join(audio_dir, filename) for filename in audio_files]
    raw_srt_paths = [os.path.splitext(path)[0] + '_RAW.srt' for path in audio_paths]
    final_srt_paths = [os.path.splitext(path)[0] + '.srt' for path in audio_paths]
    
    # Process all files in a single batch for segment-level optimization
    print(f"Generating transcriptions for all {total_files} files together")
    multiple_generate_srt_from_audio(audio_paths, raw_srt_paths)
    
    # Now that we have all the raw SRTs, process text alignment for each file in parallel
    print(f"Starting text alignment for all files")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), total_files)) as executor:
        # Prepare alignment tasks
        alignment_tasks = []
        
        for i, filename in enumerate(audio_files):
            text_path = os.path.join(text_dir, os.path.splitext(filename)[0] + '.txt')
            
            print(f"{i+1}/{total_files} - Submitting text alignment for: {filename}")
            task = executor.submit(
                iterative_sync_text_srt,
                text_path,
                raw_srt_paths[i],
                whisperTimeSync,
                i+1  # Use as file_id
            )
            alignment_tasks.append((task, i, filename))
        
        # Process results as they complete
        completed_files = []
        for task, i, filename in alignment_tasks:
            try:
                final_temp_path, temp_dir = task.result()
                
                # Move final synced SRT to destination
                os.rename(final_temp_path, final_srt_paths[i])
                
                # Clean up temporary files
                remove_steps_files(temp_dir)
                os.rmdir(temp_dir)
                
                print(f"Completed {filename}!")
                completed_files.append(filename)
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    print(f"Completed {len(completed_files)}/{total_files} files")
    print("All files processed!")

if __name__ == "__main__":
    main()
