import os
import srt
import glob
import argparse
from datetime import timedelta
from collections import Counter
import re
import statistics
import copy

# Hebrew niqqud unicode ranges to filter out
NIKUD_CHARS = ["ֱ", "ֲ", "ֳ", "ִ", "ֵ", "ֶ", "ַ", "ָ", "ׂ", "ׁ", "ֹ", "ּ", "ֻ", "ְ", "ׇ"]

def count_non_nikud_chars(text):
    """
    Count characters that are not niqqud in Hebrew text.
    This counts all characters including teamim, but excludes niqqud.
    
    Args:
        text: Hebrew text
        
    Returns:
        Number of non-niqqud characters
    """
    # Remove nikud characters
    for nikud in NIKUD_CHARS:
        text = text.replace(nikud, "")
    
    # Return the length of the remaining text
    return len(text)

def get_raw_srt_path(processed_path):
    """Convert a processed SRT path to its corresponding RAW SRT path"""
    # Replace base directory
    if "/train_data/" in processed_path:
        raw_path = processed_path.replace("/train_data/", "/train_data_RAW_SRT/")
        # Insert _RAW before the .srt extension
        raw_path = raw_path.replace(".srt", "_RAW.srt")
        return raw_path
    return None

def load_srt_file(srt_path):
    """
    Load subtitles from an SRT file.
    
    Args:
        srt_path: Path to the SRT file
        
    Returns:
        Tuple of (List of srt.Subtitle objects, error message if any)
    """
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        return list(srt.parse(srt_content)), None
    except Exception as e:
        # Extract only the error message without the content
        error_msg = str(e)
        if "unmatched content:" in error_msg:
            error_msg = error_msg.split("unmatched content:")[0].strip()
        return [], error_msg


def get_words(text):
    """Extract words from text"""
    # Remove punctuation and split into words
    return re.findall(r'\w+', text.lower())

def find_srt_files(directory, recursive=True):
    """
    Find all SRT files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of SRT file paths
    """
    if recursive:
        # Recursive search
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.srt'):
                    matches.append(os.path.join(root, filename))
        return matches
    else:
        # Non-recursive search
        return glob.glob(os.path.join(directory, "*.srt"))

def analyze_srt_files(srt_directory, analysis_type="longest", top_n=10, recursive=True, 
                     length_threshold=None, min_length_threshold=None):
    """
    Analyze SRT files in the specified directory.
    
    Args:
        srt_directory: Directory containing SRT files
        analysis_type: Type of analysis to perform
        top_n: Number of top results to return
        recursive: Whether to search recursively in subdirectories
        length_threshold: Threshold for subtitle length to include in results
        min_length_threshold: Find subtitles with non-nikud character count <= this threshold
        
    Returns:
        Tuple of (results_list, stats_dict, error_files, long_files, short_subtitles, files_with_short_subtitles)
    """
    # Find all SRT files in the directory
    srt_files = find_srt_files(srt_directory, recursive)
    
    if not srt_files:
        print(f"No SRT files found in {srt_directory}")
        return [], None, [], [], [], {}
    
    print(f"Found {len(srt_files)} SRT files to analyze")
    
    all_results = []
    word_counter = Counter()
    all_durations = []
    all_lengths = []
    all_wpm = []
    error_files = []      # List to track files with errors
    long_files = []       # List to track files with subtitles exceeding threshold
    short_subtitles = []  # List to track subtitles with few non-nikud characters
    files_with_short_subtitles = {} # Dictionary mapping file paths to lists of short subtitle indices
    
    for srt_path in srt_files:
        file_path = srt_path  # Full path
        file_name = os.path.basename(srt_path)  # Just the filename
        relative_path = os.path.relpath(srt_path, srt_directory)  # Relative path
        
        subtitles, error = load_srt_file(srt_path)
        
        if error:
            error_files.append({
                'file_path': file_path,
                'relative_path': relative_path,
                'error': error
            })
            continue
            
        file_has_long_subtitle = False
        short_subtitle_indices = []
        
        for subtitle in subtitles:
            # Count non-nikud characters instead of all characters
            length = count_non_nikud_chars(subtitle.content)
            duration = subtitle.end - subtitle.start
            duration_seconds = duration.total_seconds()
            
            # Check for short subtitles (few non-nikud characters)
            if min_length_threshold is not None and length <= min_length_threshold:
                short_subtitles.append({
                    'file': file_name,
                    'file_path': file_path,
                    'relative_path': relative_path,
                    'index': subtitle.index,
                    'content': subtitle.content,
                    'length': length,
                    'start': subtitle.start,
                    'end': subtitle.end,
                    'duration': duration
                })
                short_subtitle_indices.append(subtitle.index)
            
            # Check if this subtitle exceeds the length threshold
            if length_threshold and length > length_threshold:
                file_has_long_subtitle = True
            
            # Collect data for statistical analysis
            all_durations.append(duration_seconds)
            all_lengths.append(length)
            
            # Count words
            words = get_words(subtitle.content)
            word_count = len(words)
            
            # Calculate words per minute if duration is not zero
            wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0
            all_wpm.append(wpm)
            
            # Add to word counter for common words analysis
            word_counter.update(words)
            
            all_results.append({
                'file': file_name,
                'file_path': file_path,  # Full path
                'relative_path': relative_path,  # Relative path
                'index': subtitle.index,
                'content': subtitle.content,
                'length': length,
                'start': subtitle.start,
                'end': subtitle.end,
                'duration': duration,
                'word_count': word_count,
                'wpm': wpm
            })
        
        # Add to long files list if any subtitle exceeded the threshold
        if file_has_long_subtitle:
            long_files.append({
                'file_path': file_path,
                'relative_path': relative_path
            })
        
        # Add to files with short subtitles if any were found
        if short_subtitle_indices:
            files_with_short_subtitles[file_path] = short_subtitle_indices
    
    # Return results based on analysis type
    stats = None
    results = None
    
    if analysis_type == "longest":
        all_results.sort(key=lambda x: x['length'], reverse=True)
        results = all_results[:top_n]
    elif analysis_type == "duration":
        all_results.sort(key=lambda x: x['duration'], reverse=True)
        results = all_results[:top_n]
    elif analysis_type == "wpm":
        all_results.sort(key=lambda x: x['wpm'], reverse=True)
        results = all_results[:top_n]
    elif analysis_type == "common_words":
        results = word_counter.most_common(top_n)
    elif analysis_type == "stats":
        # Calculate and return statistics
        stats = {
            'durations': {
                'mean': statistics.mean(all_durations) if all_durations else 0,
                'median': statistics.median(all_durations) if all_durations else 0,
                'min': min(all_durations) if all_durations else 0,
                'max': max(all_durations) if all_durations else 0,
            },
            'lengths': {
                'mean': statistics.mean(all_lengths) if all_lengths else 0,
                'median': statistics.median(all_lengths) if all_lengths else 0,
                'min': min(all_lengths) if all_lengths else 0,
                'max': max(all_lengths) if all_lengths else 0,
            },
            'wpm': {
                'mean': statistics.mean(all_wpm) if all_wpm else 0,
                'median': statistics.median(all_wpm) if all_wpm else 0,
                'min': min(all_wpm) if all_wpm else 0,
                'max': max(all_wpm) if all_wpm else 0,
            }
        }
    
    return results, stats, error_files, long_files, short_subtitles, files_with_short_subtitles

def remove_short_subtitles(file_path, indices_to_remove):
    """
    Remove specified subtitles from an SRT file and adjust indices.
    
    Args:
        file_path: Path to the SRT file
        indices_to_remove: List of subtitle indices to remove
    
    Returns:
        Tuple of (success_flag, error_message)
    """
    try:
        # Load the SRT file
        subtitles, error = load_srt_file(file_path)
        if error:
            return False, f"Error loading file: {error}"
        
        if not subtitles:
            return False, "No subtitles found in the file"
        
        # Create a new list without the specified indices
        new_subtitles = []
        for subtitle in subtitles:
            if subtitle.index not in indices_to_remove:
                new_subtitles.append(subtitle)
        
        # Renumber the remaining subtitles
        for i, subtitle in enumerate(new_subtitles, 1):
            subtitle.index = i
        
        # Write the updated subtitles back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(new_subtitles))
        
        return True, f"Successfully removed {len(indices_to_remove)} subtitles from {file_path}"
    
    except Exception as e:
        return False, f"Error removing subtitles: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Analyze SRT subtitle files')
    parser.add_argument('--dir', type=str, default="/home/prj8045/train_data/Maroco-Michael-Bitton/", 
                        help='Directory containing SRT files')
    parser.add_argument('--analysis', type=str, default='longest', 
                        choices=['all', 'longest', 'duration', 'wpm', 'common_words', 'stats'],
                        help='Type of analysis to perform')
    parser.add_argument('--top', type=int, default=None, help='Number of top results to show')
    parser.add_argument('--recursive', action='store_true', help='Search for SRT files recursively in subdirectories')
    parser.add_argument('--show-content', action='store_true', help='Show subtitle content in results')
    parser.add_argument('--length-threshold', type=int, help='Report files with subtitles longer than this threshold')
    parser.add_argument('--list-error-files', action='store_true', help='List files with parsing errors')
    parser.add_argument('--min-length', type=int, default=None, 
                       help='Find subtitles with non-nikud character count <= this threshold (use 0 for empty subtitles)')
    parser.add_argument('--remove-short', action='store_true', 
                       help='Remove subtitles with few characters (specified by --min-length)')
    
    args = parser.parse_args()
    
    srt_directory = args.dir
    analysis_type = args.analysis
    top_n = args.top if args.top is not None else 10  # Default to 10 if not provided
    top_n_provided = args.top is not None  # Track if --top was explicitly provided
    recursive = args.recursive
    show_content = args.show_content
    length_threshold = args.length_threshold
    list_error_files = args.list_error_files
    min_length_threshold = args.min_length
    remove_short = args.remove_short
    
    # Verify directory exists
    if not os.path.exists(srt_directory):
        print(f"Error: Directory '{srt_directory}' does not exist")
        return
    
    print(f"Analyzing SRT files in {srt_directory}...")
    if recursive:
        print("Searching recursively in subdirectories")
    
    # Find SRT files in the directory
    srt_files = find_srt_files(srt_directory, recursive)
    if not srt_files:
        possible_locations = [
            "/home/prj8045/train_data",
            "/home/prj8045/Torah-reading-data--alignment-and-slicer",
            "/home/prj8045/Torah-reading-data--alignment-and-slicer/automatic using WhisperTimeSync"
        ]
        
        print("\nNo SRT files found. Here are some common locations to check:")
        for loc in possible_locations:
            if os.path.exists(loc):
                loc_srt_files = find_srt_files(loc, True)
                if loc_srt_files:
                    print(f"  - Found {len(loc_srt_files)} SRT files in {loc}")
                    print(f"    Try: --dir \"{loc}\"")
        
        print("\nIf you're generating SRT files using main.py, check the output location in that script.")
        print("You can also try running with --recursive to search in subdirectories")
        return
    
    # Display directory statistics
    directory_counts = {}
    for srt_path in srt_files:
        dir_name = os.path.dirname(srt_path)
        directory_counts[dir_name] = directory_counts.get(dir_name, 0) + 1
    
    print("\nFiles found by directory:")
    for dir_name, count in sorted(directory_counts.items()):
        print(f"  {dir_name}: {count} files")
    
    # Perform analyses based on selected type
    results, stats, error_files, long_files, short_subtitles, files_with_short_subtitles = analyze_srt_files(
        srt_directory, 
        analysis_type=analysis_type if analysis_type != 'all' else 'longest', 
        top_n=top_n, 
        recursive=recursive,
        length_threshold=length_threshold,
        min_length_threshold=min_length_threshold
    )
    
    # Display analysis results - only if --top was explicitly provided
    if top_n_provided and (analysis_type == 'all' or analysis_type == 'longest'):
        if results:
            print(f"\nTop {top_n} Longest Subtitles (by non-nikud character count):")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"{i+1}. File: {result['file_path']}")
                raw_path = get_raw_srt_path(result['file_path'])
                if raw_path and os.path.exists(raw_path):
                    print(f"   RAW file: {raw_path}")
                print(f"   Index: {result['index']}")
                print(f"   Length: {result['length']} non-nikud chars, Duration: {result['duration'].total_seconds():.2f}s")
                print(f"   Time: {result['start']} --> {result['end']}")
                if show_content:
                    print(f"   Content: {result['content']}")
                print("-" * 80)

    if top_n_provided and (analysis_type == 'all' or analysis_type == 'duration'):
        # Need to perform this analysis separately for 'all' mode
        if analysis_type == 'all':
            results, _, _, _, _, _ = analyze_srt_files(
                srt_directory, 
                analysis_type="duration", 
                top_n=top_n, 
                recursive=recursive,
                length_threshold=length_threshold,
                min_length_threshold=min_length_threshold
            )
            
        if results:
            print(f"\nTop {top_n} Longest Subtitles (by duration):")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"{i+1}. File: {result['file_path']}")
                raw_path = get_raw_srt_path(result['file_path'])
                if raw_path and os.path.exists(raw_path):
                    print(f"   RAW file: {raw_path}")
                print(f"   Index: {result['index']}")
                print(f"   Duration: {result['duration'].total_seconds():.2f}s, Length: {result['length']} non-nikud chars")
                print(f"   Time: {result['start']} --> {result['end']}")
                if show_content:
                    print(f"   Content: {result['content']}")
                print("-" * 80)

    if top_n_provided and (analysis_type == 'all' or analysis_type == 'wpm'):
        # Need to perform this analysis separately for 'all' mode
        if analysis_type == 'all':
            results, _, _, _, _, _ = analyze_srt_files(
                srt_directory, 
                analysis_type="wpm", 
                top_n=top_n, 
                recursive=recursive,
                length_threshold=length_threshold,
                min_length_threshold=min_length_threshold
            )
            
        if results:
            print(f"\nTop {top_n} Fastest Subtitles (by words per minute):")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"{i+1}. File: {result['file_path']}")
                raw_path = get_raw_srt_path(result['file_path'])
                if raw_path and os.path.exists(raw_path):
                    print(f"   RAW file: {raw_path}")
                print(f"   Index: {result['index']}")
                print(f"   WPM: {result['wpm']:.2f}, Words: {result['word_count']}, Duration: {result['duration'].total_seconds():.2f}s")
                print(f"   Time: {result['start']} --> {result['end']}")
                if show_content:
                    print(f"   Content: {result['content']}")
                print("-" * 80)

    if top_n_provided and (analysis_type == 'all' or analysis_type == 'common_words'):
        # Need to perform this analysis separately for 'all' mode
        if analysis_type == 'all':
            results, _, _, _, _, _ = analyze_srt_files(
                srt_directory, 
                analysis_type="common_words", 
                top_n=top_n, 
                recursive=recursive,
                length_threshold=length_threshold,
                min_length_threshold=min_length_threshold
            )
            
        if results:
            print(f"\nTop {top_n} Most Common Words:")
            print("=" * 80)
            
            for i, (word, count) in enumerate(results):
                print(f"{i+1}. '{word}': {count} occurrences")

    if analysis_type == 'all' or analysis_type == 'stats':
        # Need to perform this analysis separately for 'all' mode
        if analysis_type == 'all':
            _, stats, _, _, _, _ = analyze_srt_files(
                srt_directory, 
                analysis_type="stats", 
                recursive=recursive,
                length_threshold=length_threshold,
                min_length_threshold=min_length_threshold
            )
            
        if stats:
            print("\nSubtitle Statistics:")
            print("=" * 80)
            print("Duration Statistics (seconds):")
            print(f"  Mean: {stats['durations']['mean']:.2f}")
            print(f"  Median: {stats['durations']['median']:.2f}")
            print(f"  Min: {stats['durations']['min']:.2f}")
            print(f"  Max: {stats['durations']['max']:.2f}")
            
            print("\nLength Statistics (non-nikud characters):")
            print(f"  Mean: {stats['lengths']['mean']:.2f}")
            print(f"  Median: {stats['lengths']['median']:.2f}")
            print(f"  Min: {stats['lengths']['min']:.2f}")
            print(f"  Max: {stats['lengths']['max']:.2f}")
            
            print("\nWords Per Minute Statistics:")
            print(f"  Mean: {stats['wpm']['mean']:.2f}")
            print(f"  Median: {stats['wpm']['median']:.2f}")
            print(f"  Min: {stats['wpm']['min']:.2f}")
            print(f"  Max: {stats['wpm']['max']:.2f}")
    
    # Display files with errors if requested or if there are errors
    if list_error_files and error_files:
        print("\nFiles with parsing errors:")
        print("=" * 80)
        for i, error_file in enumerate(error_files):
            print(f"{i+1}. {error_file['file_path']}")
            raw_path = get_raw_srt_path(error_file['file_path'])
            if raw_path and os.path.exists(raw_path):
                print(f"   RAW file: {raw_path}")
            print(f"   Error: {error_file['error']}")
            print("-" * 80)
    
    # Display files with subtitles exceeding length threshold
    if length_threshold and long_files:
        print(f"\nFiles with subtitles longer than {length_threshold} non-nikud characters:")
        print("=" * 80)
        for i, long_file in enumerate(long_files):
            print(f"{i+1}. {long_file['file_path']}")
            raw_path = get_raw_srt_path(long_file['file_path'])
            if raw_path and os.path.exists(raw_path):
                print(f"   RAW file: {raw_path}")
        print(f"\nTotal: {len(long_files)} files with long subtitles")
    
    # Display short subtitles if requested
    if min_length_threshold is not None and short_subtitles:
        print(f"\nSubtitles with {min_length_threshold} or fewer non-nikud characters:")
        print("=" * 80)
        for i, short in enumerate(short_subtitles):
            print(f"{i+1}. File: {short['file_path']}")
            raw_path = get_raw_srt_path(short['file_path'])
            if raw_path and os.path.exists(raw_path):
                print(f"   RAW file: {raw_path}")
            print(f"   Index: {short['index']}")
            print(f"   Length: {short['length']} non-nikud chars")
            print(f"   Time: {short['start']} --> {short['end']}")
            print(f"   Duration: {short['duration'].total_seconds():.2f}s")
            # Always show content for short subtitles since they're very brief
            print(f"   Content: '{short['content']}'")
            print("-" * 80)
        print(f"\nTotal: {len(short_subtitles)} short subtitles found in {len(files_with_short_subtitles)} files")
        
        # If option to remove short subtitles is enabled, ask for confirmation
        if remove_short and files_with_short_subtitles:
            print("\nFound files with short subtitles that can be removed:")
            for i, (file_path, indices) in enumerate(files_with_short_subtitles.items()):
                print(f"{i+1}. {file_path} - {len(indices)} subtitle(s) to remove")
                raw_path = get_raw_srt_path(file_path)
                if raw_path and os.path.exists(raw_path):
                    print(f"   RAW file: {raw_path}")
            
            # Ask for confirmation
            confirmation = input("\nDo you want to remove these short subtitles? (yes/no): ").strip().lower()
            if confirmation in ['yes', 'y']:
                print("\nRemoving short subtitles...")
                
                success_count = 0
                error_count = 0
                
                for file_path, indices in files_with_short_subtitles.items():
                    success, message = remove_short_subtitles(file_path, indices)
                    
                    if success:
                        print(f"✓ {file_path}: Removed {len(indices)} subtitle(s)")
                        success_count += 1
                    else:
                        print(f"✗ {file_path}: {message}")
                        error_count += 1
                
                print(f"\nRemoval completed: {success_count} files updated, {error_count} errors")
            else:
                print("Operation cancelled. No changes were made.")

if __name__ == "__main__":
    main()
