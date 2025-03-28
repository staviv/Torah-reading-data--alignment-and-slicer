import os
import threading
import concurrent.futures
from audio_to_srt import multiple_generate_srt_from_audio, initialize_models
from create_steps_files import create_steps_files, remove_steps_files
import subprocess
import shutil

# Define the base directory containing the audio files and subdirectories
base_dir = "/home/prj8045/train_data/"
text_dir = "/home/prj8045/train_data/text/"
raw_srt_storage_base = "/home/prj8045/train_data_RAW_SRT"

whisperTimeSync = "/home/prj8045/Torah-reading-data--alignment-and-slicer/automatic using WhisperTimeSync/WhisperTimeSync/distrib/WhisperTimeSync.jar"

# Lock for synchronizing temp directory creation and cleanup
temp_dir_lock = threading.Lock()

# Create a parent temp directory
TEMP_PARENT_DIR = "temp"

def clean_temp_directory(temp_dir):
    """clean up temporary directory"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning temp directory {temp_dir}: {e}")


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

def find_text_file(audio_file_path, base_dir, text_dir):
    """
    Find the corresponding text file for an audio file
    Try:
    1. Look in the same directory as the audio file
    2. Look in the text_dir with the same basename
    """
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    text_file_name = base_name + '.txt'
    
    # Option 1: Same directory as audio file
    text_path_same_dir = os.path.join(os.path.dirname(audio_file_path), text_file_name)
    if os.path.exists(text_path_same_dir):
        return text_path_same_dir
    
    # Option 2: In the text_dir
    text_path_text_dir = os.path.join(text_dir, text_file_name)
    if os.path.exists(text_path_text_dir):
        return text_path_text_dir
    
    # If we get here, no text file was found
    return None

def move_raw_srt_to_storage(raw_srt_path, audio_path, base_dir, raw_storage_base):
    """
    Move the RAW SRT file to its corresponding location in the RAW_SRT_train_data directory
    
    Arguments:
    raw_srt_path -- Path to the RAW SRT file
    audio_path -- Path to the original audio file
    base_dir -- Base directory of the original audio files
    raw_storage_base -- Base directory for storing RAW SRT files
    """
    try:
        # Determine the relative path from the base directory
        rel_path = os.path.relpath(os.path.dirname(audio_path), base_dir)
        
        # Create the target directory
        target_dir = os.path.join(raw_storage_base, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Get the target path for the RAW SRT file
        target_raw_srt_path = os.path.join(target_dir, os.path.basename(raw_srt_path))
        
        # Move the RAW SRT file
        shutil.move(raw_srt_path, target_raw_srt_path)
        print(f"Moved RAW SRT file to: {target_raw_srt_path}")
        
    except Exception as e:
        print(f"Error moving RAW SRT file {raw_srt_path}: {e}")

def main():
    """
    Main function to process all audio files at once for optimal GPU utilization
    """
    # Initialize models once before processing any files
    initialize_models()
    
    # Create the temp parent directory
    os.makedirs(TEMP_PARENT_DIR, exist_ok=True)
    
    # Ensure RAW SRT storage base exists
    os.makedirs(raw_srt_storage_base, exist_ok=True)
    
    # Find all audio files in all subdirectories
    audio_files_to_process = []
    file_count = 0
    skipped_count = 0
    
    # Walk through all subdirectories
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                file_count += 1
                full_path = os.path.join(root, file)
                # Check if SRT file already exists
                srt_path = os.path.splitext(full_path)[0] + '.srt'
                if os.path.exists(srt_path):
                    skipped_count += 1
                    print(f"Skipping {file} - SRT file already exists")
                    continue
                
                # Find corresponding text file
                text_path = find_text_file(full_path, base_dir, text_dir)
                if text_path is None:
                    print(f"Warning: No text file found for {file}. Skipping.")
                    continue
                
                audio_files_to_process.append((full_path, text_path))
    
    total_files = len(audio_files_to_process)
    
    print(f"Found {file_count} audio files total")
    print(f"Skipping {skipped_count} files that already have SRT files")
    print(f"Processing {total_files} files")
    
    if total_files == 0:
        print("No files need processing. Exiting.")
        return
    
    print(f"Processing all files together for optimal GPU batch utilization")
    
    # Create paths for all files
    audio_paths = [file_info[0] for file_info in audio_files_to_process]
    text_paths = [file_info[1] for file_info in audio_files_to_process]
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
        
        for i, (audio_path, text_path) in enumerate(audio_files_to_process):
            filename = os.path.basename(audio_path)
            print(f"{i+1}/{total_files} - Submitting text alignment for: {filename}")
            task = executor.submit(
                iterative_sync_text_srt,
                text_path,
                raw_srt_paths[i],
                whisperTimeSync,
                i+1  # Use as file_id
            )
            alignment_tasks.append((task, i, filename, audio_paths[i], raw_srt_paths[i]))
        
        # Process results as they complete
        completed_files = []
        for task, i, filename, audio_path, raw_srt_path in alignment_tasks:
            try:
                final_temp_path, temp_dir = task.result()
                
                # Move final synced SRT to destination
                os.rename(final_temp_path, final_srt_paths[i])
                
                # Move RAW SRT file to storage location
                move_raw_srt_to_storage(raw_srt_path, audio_path, base_dir, raw_srt_storage_base)
                
                # Clean up temporary files
                clean_temp_directory(temp_dir)
                
                print(f"Completed {filename}!")
                completed_files.append(filename)
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    print(f"Completed {len(completed_files)}/{total_files} files")
    print("All files processed!")

if __name__ == "__main__":
    main()
