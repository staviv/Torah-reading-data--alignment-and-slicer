import os
import shutil
import glob

def find_and_remove_files_with_phrase(raw_srt_base_dir, base_dir, phrase="תודה רבה"):
    """
    Find RAW SRT files containing the specified phrase and remove both the RAW file
    and its corresponding final SRT file.
    
    Args:
        raw_srt_base_dir: Base directory for RAW SRT files (/home/prj8045/train_data_RAW_SRT)
        base_dir: Base directory for final SRT files (/home/prj8045/train_data/)
        phrase: Phrase to search for in RAW SRT files
    """
    # Track files removed
    raw_files_removed = []
    final_files_removed = []
    errors = []
    
    # Find all RAW SRT files
    raw_srt_files = []
    for root, _, files in os.walk(raw_srt_base_dir):
        for file in files:
            if file.endswith('_RAW.srt'):
                raw_srt_files.append(os.path.join(root, file))
    
    print(f"Found {len(raw_srt_files)} RAW SRT files to check")
    
    # Check each RAW SRT file for the problematic phrase
    problematic_files = []
    for raw_srt_path in raw_srt_files:
        try:
            with open(raw_srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if phrase in content:
                    problematic_files.append(raw_srt_path)
        except Exception as e:
            errors.append(f"Error reading {raw_srt_path}: {e}")
    
    print(f"Found {len(problematic_files)} problematic RAW SRT files containing '{phrase}'")
    
    # Process each problematic file
    for raw_srt_path in problematic_files:
        try:
            # 1. Determine the relative path from raw_srt_base_dir
            rel_path = os.path.relpath(os.path.dirname(raw_srt_path), raw_srt_base_dir)
            
            # 2. Get the original base filename (without _RAW.srt)
            raw_basename = os.path.basename(raw_srt_path)
            base_filename = raw_basename.replace('_RAW.srt', '')
            
            # 3. Construct path to final SRT file
            final_srt_path = os.path.join(base_dir, rel_path, f"{base_filename}.srt")
            
            # 4. Remove both files
            if os.path.exists(raw_srt_path):
                os.remove(raw_srt_path)
                raw_files_removed.append(raw_srt_path)
                print(f"Removed RAW SRT file: {raw_srt_path}")
                
            if os.path.exists(final_srt_path):
                os.remove(final_srt_path)
                final_files_removed.append(final_srt_path)
                print(f"Removed final SRT file: {final_srt_path}")
            else:
                print(f"Final SRT file not found: {final_srt_path}")
                
        except Exception as e:
            errors.append(f"Error processing {raw_srt_path}: {e}")
    
    # Summary
    print("\nSummary:")
    print(f"Checked {len(raw_srt_files)} RAW SRT files")
    print(f"Found {len(problematic_files)} files containing '{phrase}'")
    print(f"Removed {len(raw_files_removed)} RAW SRT files")
    print(f"Removed {len(final_files_removed)} final SRT files")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"- {error}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")
    
    return raw_files_removed, final_files_removed, errors

# Example usage
if __name__ == "__main__":
    raw_srt_base_dir = "/home/prj8045/train_data_RAW_SRT"
    base_dir = "/home/prj8045/train_data"
    find_and_remove_files_with_phrase(raw_srt_base_dir, base_dir)
