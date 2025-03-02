import yt_dlp
import google.generativeai as genai
import json
import os
import sys
from typing import Dict, Set, Optional, List, Tuple
from get_all_aliyot_from_sefaria import parsha_names # parsha_names is a list of all the parshiot in the Torah
# from generate_parasha_variants import clean_variant
from parasha_matcher import ParashaMatcher

GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

# Define valid combined parashot with & separator
VALID_COMBINED_PARASHOT = [
    "Vayakhel&Pekudei",
    "Tazria&Metzora",
    "AchreiMot&Kedoshim",
    "Behar&Bechukotai",
    "Chukat&Balak",
    "Matot&Masei",
    "Nitzavim&Vayeilech"
]

def get_api_key():
    try:
        return os.environ["GEMINI_API_KEY"]
    except KeyError:
        api_key = input("Please enter your Gemini API key: ").strip()
        if not api_key:
            print("Error: No API key provided")
            sys.exit(1)
        return api_key

def get_video_metadata(link):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(link, download=False)
            return {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'channel': info.get('channel', ''),
                # 'upload_date': info.get('upload_date', ''),
                # 'duration': info.get('duration', 0),
                # 'view_count': info.get('view_count', 0),
            }
        except Exception as e:
            print(f"Warning: Could not extract video metadata: {str(e)}")
            return {}

def clean_video_url(url):
    """Remove playlist parameters from video URLs."""
    if 'youtube.com/watch?' in url:
        # Extract video ID
        video_id = None
        for param in url.split('?')[1].split('&'):
            if param.startswith('v='):
                video_id = param.split('=')[1]
                break
        
        if video_id:
            return f'https://www.youtube.com/watch?v={video_id}'
    
    return url

def is_playlist(url):
    return 'playlist?list=' in url  # Modified to only match proper playlist URLs

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'ignore_errors': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            playlist_dict = ydl.extract_info(playlist_url, download=False)
            return [entry['url'] for entry in playlist_dict['entries'] if entry is not None]
        except Exception as e:
            print(f"Error extracting playlist: {str(e)}")
            return []

def get_last_dataset_name():
    try:
        with open("last_dataset.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_dataset_name(dataset_name):
    with open("last_dataset.txt", "w") as f:
        f.write(dataset_name)

def is_valid_parasha(parasha_name):
    """Validate single or combined parashot."""
    # Check if it's a single valid parasha
    if parasha_name in parsha_names:
        return True
    
    # Check if it's a valid combined parasha
    if parasha_name in VALID_COMBINED_PARASHOT:
        return True
    
    return False

def clean_variant(name: str) -> str:
    """Clean a variant name to lowercase alphanumeric."""
    if not name or not isinstance(name, str):
        return ""
    return name.lower().strip()

def setup_llm(api_key):
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }
        
        system_instruction = f"""
        You are a Torah reading assistant. When given a YouTube video's metadata, analyze it and return a JSON object.
        
        IMPORTANT RULES FOR is_parasha FIELD:
        Set is_parasha to TRUE if you can understand which parasha is being read in the video and it is NOT something else like a Torah lesson, haftarah, megillah, maftir, etc.
        
        In parasha FIELD you must return either:
        1. Exactly one of the valid parasha names from this list: {parsha_names}
        2. OR a combined parasha from this list: {VALID_COMBINED_PARASHOT}
        
        The response must include the following fields:
        
        {{
            "is_parasha": boolean (true ONLY if all conditions above are met),
            "parasha": string (must be exactly one of the valid parasha names, a valid combined parasha, or null if not a weekly torah reading),
            "aliyah": string (either "all" for full parasha or "1"-"7" for specific aliyah),
            "dataset_name": string (suggested name for the dataset, or null if not applicable),
            "confidence": float (0-1, how confident you are in the prediction)
        }}

        Example outputs:
        {{"is_parasha": true, "parasha": "Bereshit", "aliyah": "1", "dataset_name": "Nusach-Ashkenaz-david-Goldberg", "confidence": 0.95}}
        {{"is_parasha": true, "parasha": "Vayakhel&Pekudei", "aliyah": "all", "dataset_name": "Nusach-Ashkenaz-david-Goldberg", "confidence": 0.95}}
        {{"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": "Nusach-Yerushalmi-Avi-Levi", "confidence": 0.9}} <- Not a Torah reading
        {{"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": null, "confidence": 0.8}} <- Torah lesson but not reading
        {{"is_parasha": false, "parasha": "Bereshit", "aliyah": null, "dataset_name": null, "confidence": 0.7}} <- Mentions parasha but not reading
        """
        
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
    except Exception as e:
        print(f"Error setting up LLM: {str(e)}")
        sys.exit(1)

def clean_llm_json_response(response_text):
    """Extract and parse JSON from LLM response, handling code block markers."""
    # Remove code block markers if present
    start = response_text.find("```json") + 7  # Skip past ```json
    if start < 7:  # No ```json found, try without json marker
        start = response_text.find("```") + 3
        if start < 3:  # No ``` at all
            start = 0
    
    end = response_text.find("```", start)
    cleaned_text = response_text[start:end].strip() if end != -1 else response_text.strip()
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw text: {cleaned_text}")
        return None

def get_llm_suggestion(model, link, quiet=False):
    metadata = get_video_metadata(link)
    prompt = f"""
    Analyze this YouTube video metadata to identify the Torah portion(s) being read:
    Title: {metadata.get('title', '')}
    Description: {metadata.get('description', '')} 
    Channel: {metadata.get('channel', '')}
    
    IMPORTANT: Make a clear decision about which parasha is being read.
    
    If you detect that two Torah portions are read together (like Vayakhel and Pekudei), 
    you MUST use the format "Parasha1&Parasha2" with the & symbol between them.
    Use the combined format ONLY if you're certain both portions are being read together.
    
    Valid single parashot: {parsha_names}
    
    Valid combined parashot (use EXACTLY these spellings with the & symbol):
    {VALID_COMBINED_PARASHOT}
    
    Use the EXACT spelling from these lists. Do not create your own combinations.
    """
    response = model.generate_content(prompt)
    result = clean_llm_json_response(response.text)
    if not result:
        return {}
    
    # Print metadata and confidence only if not in quiet mode
    if not quiet:
        print("\nVideo metadata:")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Channel: {metadata.get('channel', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    return result

def get_input_with_suggestion(prompt, suggestion, validator=None):
    suggestion_str = str(suggestion) if suggestion else ""
    user_input = input(f"{prompt} [{suggestion_str}]: ").strip()
    if not user_input and suggestion_str:
        return suggestion
    if validator:
        while not validator(user_input):
            user_input = input(f"Invalid input. {prompt}: ")
    return user_input

def download_audio(link, output_path, start_time=None, end_time=None, quiet=False):
    # Remove .mp3 extension as it will be added by yt-dlp
    if output_path.endswith('.mp3'):
        output_path = output_path[:-4]

    # Check if file already exists
    if os.path.exists(output_path + '.mp3'):
        if not quiet:
            print(f"\nWarning: File already exists at {output_path}.mp3")
            if not input("Do you want to overwrite it? (y/n): ").lower().startswith('y'):
                print("Skipping download...")
                return False
        else:
            return False

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': output_path,  # Extension will be added automatically
    }

    # Add quiet options when in quiet mode
    if quiet:
        ydl_opts.update({
            'quiet': True,
            'no_warnings': True,
            'no_color': True
        })

    # Add download range if times are specified
    if start_time or end_time:
        download_range = ''
        if start_time:
            download_range += start_time
        download_range += '-'
        if end_time:
            download_range += end_time
        ydl_opts['download_ranges'] = lambda info: [[download_range]]
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    return True

def print_parshiot():
    # Show aliyah range for the selected parsha and get confirmation
    print("\nHere are all parshiot with their numbers:")
    for i, p in enumerate(parsha_names, 1):
        print(f"{p}: {i}", end="  ")
        if i % 4 == 0:  # Print 4 parshiot per line
            print()
    print("\n\nValid combined parashot:")
    for cp in VALID_COMBINED_PARASHOT:
        print(cp)

def select_parasha(suggestion):
    """Enhanced parasha selection that handles combined parashot."""
    print_parshiot()
    
    suggested_parasha = suggestion.get("parasha")
    
    if suggested_parasha in parsha_names:
        suggested_parsha_num = parsha_names.index(suggested_parasha) + 1
        print(f"\nSuggested parasha: {suggested_parasha}")
        use_suggestion = input("Use this suggested parasha? (y/n): ").lower().startswith('y')
        if use_suggestion:
            return suggested_parasha
    elif suggested_parasha in VALID_COMBINED_PARASHOT:
        print(f"\nSuggested combined parashot: {suggested_parasha}")
        use_suggestion = input("Use this combined parasha? (y/n): ").lower().startswith('y')
        if use_suggestion:
            return suggested_parasha
    
    # Ask if user wants to enter a single parasha or combined parashot
    choice = input("\nEnter '1' for single parasha or '2' for combined parashot: ")
    
    if choice == "1":
        # Single parasha selection logic
        suggested_num = ""
        if suggested_parasha in parsha_names:
            suggested_num = parsha_names.index(suggested_parasha) + 1
            
        parsha_num = int(get_input_with_suggestion(
            "Enter the number of the parsha: ", 
            suggested_num, 
            lambda x: x.isdigit() and 1 <= int(x) <= len(parsha_names)))
        return parsha_names[parsha_num - 1]
    else:
        # Combined parashot selection
        print("\nChoose a combined parasha:")
        for i, cp in enumerate(VALID_COMBINED_PARASHOT, 1):
            print(f"{i}. {cp}")
        
        combined_num = int(get_input_with_suggestion(
            "Enter the number of the combined parasha: ", 
            "", 
            lambda x: x.isdigit() and 1 <= int(x) <= len(VALID_COMBINED_PARASHOT)))
        
        return VALID_COMBINED_PARASHOT[combined_num - 1]

class EnhancedParashaMatcher(ParashaMatcher):
    def match_parasha_name(self, input_name):
        """Enhanced matching that handles combined parashot."""
        if not input_name or not isinstance(input_name, str):
            return None, 0, False
            
        # First check if it's a valid combined parasha name
        for combined in VALID_COMBINED_PARASHOT:
            if combined.lower() == input_name.lower():
                return combined, 1.0, True
        
        # Check if it might be a combined parasha with different separators
        if "-" in input_name or " and " in input_name or "&" in input_name or "+" in input_name or " " in input_name:
            # Replace common separators with standardized format
            normalized = input_name.replace(" and ", "&").replace("-", "&").replace("+", "&")
            normalized = normalized.replace(" ", "&")  # Also try spaces as separators
            parts = [p.strip() for p in normalized.split("&") if p.strip()]
            
            if len(parts) == 2:
                # Try to match each part separately
                match1, conf1, exact1 = self._match_single_parasha(parts[0])
                match2, conf2, exact2 = self._match_single_parasha(parts[1])
                
                if match1 and match2:
                    # Check if this is a valid combination
                    potential_combo = f"{match1}&{match2}"
                    for valid_combo in VALID_COMBINED_PARASHOT:
                        if valid_combo.lower() == potential_combo.lower():
                            avg_confidence = (conf1 + conf2) / 2
                            is_exact = exact1 and exact2
                            return valid_combo, avg_confidence, is_exact
                    
                    # Check the reverse order too
                    potential_combo_reversed = f"{match2}&{match1}"
                    for valid_combo in VALID_COMBINED_PARASHOT:
                        if valid_combo.lower() == potential_combo_reversed.lower():
                            avg_confidence = (conf1 + conf2) / 2
                            is_exact = exact1 and exact2
                            return valid_combo, avg_confidence, is_exact
        
        # Fall back to single parasha matching
        return super().match_parasha_name(input_name)
    
    def _match_single_parasha(self, name):
        """Match a single parasha name."""
        # Use the original matching logic from ParashaMatcher
        return super().match_parasha_name(name)

def match_parasha_name(model, suggested_name):
    """Match a suggested parasha name to a valid parasha or combined parasha."""
    # Path of current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    matcher = EnhancedParashaMatcher(f"{current_dir}/parasha_variants.json")
    
    if (not suggested_name) or (is_valid_parasha(suggested_name)):
        return suggested_name
    
    matched_name, confidence, is_exact = matcher.match_parasha_name(suggested_name)
    if matched_name:
        match_type = "exact" if is_exact else f"fuzzy (confidence: {confidence:.2f})"
        print(f"\nMatched '{suggested_name}' to '{matched_name}' using {match_type} matching")
        return matched_name
    
    # If no match found, try splitting the name and matching parts individually
    if suggested_name and isinstance(suggested_name, str):
        words = suggested_name.split()
        if len(words) >= 2:
            print(f"\nTrying to match individual words in '{suggested_name}'...")
            
            # Try to match each word individually
            matched_parashas = []
            for word in words:
                if word.lower() in ['and', '&', 'parshat', 'parasha', 'parashat', 'parsha', 'portion', 'torah']:
                    continue
                    
                word_match, word_conf, word_exact = matcher.match_parasha_name(word)
                if word_match and word_conf > 0.7:
                    matched_parashas.append((word_match, word_conf))
            
            # If we found exactly two parashas, check if they form a valid combination
            if len(matched_parashas) == 2:
                p1, conf1 = matched_parashas[0]
                p2, conf2 = matched_parashas[1]
                
                potential_combo = f"{p1}&{p2}"
                if potential_combo in VALID_COMBINED_PARASHOT:
                    print(f"\nFound valid combined parasha: {potential_combo}")
                    return potential_combo
                
                # Try reverse order
                potential_combo = f"{p2}&{p1}"
                if potential_combo in VALID_COMBINED_PARASHOT:
                    print(f"\nFound valid combined parasha: {potential_combo}")
                    return potential_combo
    
    print(f"\nWarning: Could not match '{suggested_name}' to any known variants")
    return None

def process_single_video(model, link, dataset_name=None):
    suggestion = get_llm_suggestion(model, link)
    print(f"\nSuggestion: {suggestion}")
    
    # Check if this is a weekly parasha reading
    is_parasha = suggestion.get("is_parasha", False)
    if not is_parasha:
        print("\nWarning: This content does not appear to be a weekly Torah portion reading")
        if not input("\nDo you want to continue anyway? (y/n): ").lower().startswith('y'):
            return False, dataset_name

    suggested_parasha = suggestion.get("parasha")
    if not is_valid_parasha(suggested_parasha):
        print(f"\nSuggested parasha name: {suggested_parasha}")
        print("\nThis parasha name is not in the valid list")
        print("\nTrying to match suggested parasha name to valid list...")
        matched_parasha = match_parasha_name(model, suggested_parasha)
        if matched_parasha:
            suggestion["parasha"] = matched_parasha
        else:
            print("\nWarning: Could not match parasha name to valid list")
            if not input("Do you want to continue processing this current video (otherwise it will be skipped)? (y/n): ").lower().startswith('y'):
                return False, dataset_name

    # Use the enhanced parasha selection function
    parsha = select_parasha(suggestion)
    
    aliyah = get_input_with_suggestion(
        "Enter aliyah number (1-7) or 'all' for full parasha: ", 
        suggestion.get("aliyah"), 
        lambda x: x == "all" or (x.isdigit() and 1 <= int(x) <= 7))

    if dataset_name is None:
        suggested_dataset = suggestion.get("dataset_name")
        last_dataset_name = get_last_dataset_name()
        
        if last_dataset_name:
            print(f"\nLast used dataset name: {last_dataset_name}")
            use_last = input("Use this name? (y/n): ").lower().startswith('y')
            if use_last:
                dataset_name = last_dataset_name
        
        if not dataset_name and suggested_dataset:
            print(f"\nSuggested dataset name: {suggested_dataset}")
            user_input = input(f"Press Enter to use this name, or type a different name: ").strip()
            dataset_name = suggested_dataset if not user_input else user_input
        
        if not dataset_name:
            dataset_name = input("Enter the name of the dataset: ")
    else:
        # Show current dataset name and ask if user wants to change it
        print(f"\nCurrent dataset name: {dataset_name}")
        if not input("Continue with this dataset name? (y/n): ").lower().startswith('y'):
            dataset_name = input("Enter new dataset name: ")

    dataset_dir = f"/home/prj8045/data/{dataset_name}"
    audio_path = f"{dataset_dir}/{parsha}-{aliyah}.mp3"
    
    # Add time selection before download
    start_time = input("\nEnter start time (optional, format MM:SS or HH:MM:SS, press Enter to skip): ").strip()
    end_time = input("Enter end time (optional, format MM:SS or HH:MM:SS, press Enter to skip): ").strip()
    
    print(f"\nDownloading to {audio_path}...")
    if not download_audio(link, audio_path, start_time, end_time):
        return False, dataset_name
    print("Download complete")
    
    return True, dataset_name

def has_playlist_param(url):
    """Check if URL contains playlist parameter."""
    return 'list=' in url

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    for param in url.split('?')[1].split('&'):
        if param.startswith('list='):
            return param.split('=')[1]
    return None

def get_playlist_url(playlist_id):
    """Convert playlist ID to playlist URL."""
    return f'https://www.youtube.com/playlist?list={playlist_id}'

def read_video_links_from_file(file_path):
    """Read video and playlist links from a text file."""
    links = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Process the URL in case it's a playlist
                    processed_url = process_playlist_url(line)
                    if is_playlist(processed_url):
                        # If it's a playlist, get all video URLs
                        playlist_videos = get_playlist_videos(processed_url)
                        links.extend(playlist_videos)
                    else:
                        # If it's a single video, add it directly
                        links.append(line)
    except Exception as e:
        print(f"Error reading video links file: {str(e)}")
    return links

def save_unmatched_link(link, dataset_name, reason=""):
    """Save unmatched video link to a file with reason for failure."""
    os.makedirs('unmatched_videos', exist_ok=True)
    filename = f'unmatched_videos/unmatched_{dataset_name}.txt'
    
    # Check if link already exists in file to avoid duplicates
    existing_links = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if '#' in line:
                    existing_link = line.split('#')[0].strip()
                    existing_links.add(existing_link)
    
    # Only append if link isn't already in the file
    if link not in existing_links:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"{link} # {reason}\n")

def process_single_video_automatic(model, link, dataset_name):
    """Process a single video automatically without user interaction."""
    suggestion = get_llm_suggestion(model, link, quiet=True)
    
    if not suggestion.get("is_parasha", False):
        save_unmatched_link(link, dataset_name, "Not detected as Torah reading")
        return False, dataset_name

    suggested_parasha = suggestion.get("parasha")
    if not is_valid_parasha(suggested_parasha):
        matched_parasha = match_parasha_name(model, suggested_parasha)
        if matched_parasha:
            suggestion["parasha"] = matched_parasha
        else:
            save_unmatched_link(link, dataset_name, f"Could not match parasha name: {suggested_parasha}")
            return False, dataset_name

    parsha = suggestion["parasha"]
    aliyah = suggestion.get("aliyah", "all")
    
    dataset_dir = f"/home/prj8045/data/{dataset_name}"
    audio_path = f"{dataset_dir}/{parsha}-{aliyah}.mp3"
    
    if not download_audio(link, audio_path, quiet=True):
        save_unmatched_link(link, dataset_name, "Download failed or file already exists")
        return False, dataset_name
    
    return True, dataset_name

def process_videos_automatic(model, videos, dataset_name):
    """Process multiple videos automatically."""
    successes = 0
    total = len(videos)
    
    for i, video_url in enumerate(videos, 1):
        print(f"\nProcessing video {i}/{total}: {video_url}")
        success, _ = process_single_video_automatic(model, video_url, dataset_name)
        if success:
            successes += 1
    
    unmatched_file = f'unmatched_videos/unmatched_{dataset_name}.txt'
    if os.path.exists(unmatched_file):
        print(f"\nUnmatched videos were saved to: {unmatched_file}")
    print(f"\nProcessing complete. Successfully processed {successes}/{total} videos.")
    return dataset_name

def get_playlist_metadata(playlist_url):
    """Get metadata for a YouTube playlist."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            return {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'channel': info.get('uploader', ''),
                'video_count': len(info.get('entries', [])),
                'first_video_title': info.get('entries', [{}])[0].get('title', ''),
                'Channel of the first video': info.get('entries', [{}])[0].get('uploader', ''),
            }
        except Exception as e:
            print(f"Warning: Could not extract playlist metadata: {str(e)}")
            return {}

def get_dataset_name_suggestion(model, playlist_metadata):
    """Get dataset name suggestion from LLM based on playlist metadata."""
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 256,
    }

    system_instruction = """
    You are a Torah reading dataset name generator.
    You must suggest dataset names in the following format:
    Nusach-[Type]-[Reader-Name] (e.g. Nusach-Ashkenaz-david-Goldberg)
    
    Return only the suggested name, nothing else.
    No explanations or additional text.
    """
    
    prompt = f"""
    Based on this playlist information:
    Title: {playlist_metadata.get('title', '')}
    Description: {playlist_metadata.get('description', '')}
    Channel: {playlist_metadata.get('channel', '')}
    Number of videos: {playlist_metadata.get('video_count', 0)}
    Title of the first video: {playlist_metadata.get('first_video_title', '')}
    Channel of the first video: {playlist_metadata.get('Channel of the first video', '')}
    """
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config=generation_config,
        system_instruction=system_instruction
    )
    
    try:
        response = model.generate_content(prompt)
        suggested_name = response.text.strip()
        if suggested_name.startswith('"') and suggested_name.endswith('"'):
            suggested_name = suggested_name[1:-1]
        return suggested_name
    except Exception as e:
        print(f"Error getting dataset name suggestion: {str(e)}")
        return None

def process_playlist_url(url):
    """Process URL that might contain both video and playlist information."""
    if 'youtube.com/watch?' in url and has_playlist_param(url):
        playlist_id = extract_playlist_id(url)
        return get_playlist_url(playlist_id)
    return url

def main():
    api_key = get_api_key()
    model = setup_llm(api_key)
    
    last_dataset_name = get_last_dataset_name()
    dataset_name = None

    while True:
        print("\nChoose an option:")
        print("1. Process videos interactively")
        print("2. Process playlist automatically")
        print("3. Process video links file automatically")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ").lower()
        
        if choice == 'q':
            break
            
        if choice not in ['1', '2', '3']:
            print("Invalid choice")
            continue

        if choice == '1':
            # Original interactive processing
            link = input("\nEnter the YouTube video/playlist link (or 'q' to quit): ")
            if link.lower() == 'q':
                break
                
            # Check if URL contains both video and playlist
            if 'youtube.com/watch?' in link and has_playlist_param(link):
                playlist_id = extract_playlist_id(link)
                choice = input("\nThis URL contains both video and playlist. Do you want to:\n"
                             "1. Process just this video\n"
                             "2. Process the entire playlist\n"
                             "Choose (1/2): ").strip()
                
                if choice == '2':
                    link = get_playlist_url(playlist_id)
                    print(f"\nSwitching to playlist mode: {link}")
                else:
                    link = clean_video_url(link)
                    print(f"\nProcessing single video: {link}")
            else:
                # Clean the URL if it's a video
                cleaned_link = clean_video_url(link)
                if cleaned_link != link:
                    print(f"Cleaned URL: {cleaned_link}")
                    link = cleaned_link

            if is_playlist(link):
                videos = get_playlist_videos(link)
                print(f"\nFound {len(videos)} videos in playlist")
                
                # Ask for dataset name once for the entire playlist
                if dataset_name is None:
                    if last_dataset_name:
                        use_last = input(f"Use last dataset name '{last_dataset_name}'? (y/n): ").lower() == 'y'
                        dataset_name = last_dataset_name if use_last else input("Enter the name of the dataset: ")
                    else:
                        dataset_name = input("Enter the name of the dataset: ")
                
                for video_url in videos:
                    print(f"\nProcessing video: {video_url}")
                    success, dataset_name = process_single_video(model, video_url, dataset_name)
                    if not success:
                        print("Skipping to next video...")
                        continue
                    
            else:  # Single video
                success, dataset_name = process_single_video(model, link, dataset_name)
                if not success:
                    continue

        elif choice == '2':
            # Get playlist URL first
            playlist_url = input("Enter the YouTube playlist URL: ")
            playlist_url = process_playlist_url(playlist_url)
            
            if not is_playlist(playlist_url):
                print("Error: Not a valid playlist URL")
                continue
            
            # First check if there's a last used dataset name
            if last_dataset_name:
                print(f"\nLast used dataset name: {last_dataset_name}")
                use_last = input("Use this name? (y/n): ").lower().startswith('y')
                if use_last:
                    dataset_name = last_dataset_name
            
            # Only get playlist metadata and suggestion if user doesn't want to use last name
            if not dataset_name:
                # Get playlist metadata and suggest dataset name
                playlist_metadata = get_playlist_metadata(playlist_url)
                if playlist_metadata:
                    suggested_name = get_dataset_name_suggestion(model, playlist_metadata)
                    if suggested_name:
                        print(f"\nSuggested dataset name based on playlist: {suggested_name}")
                        use_suggestion = input("Use this name? (y/n): ").lower().startswith('y')
                        if use_suggestion:
                            dataset_name = suggested_name
                        else:
                            dataset_name = input("Enter the name of the dataset: ")
                    else:
                        dataset_name = input("Enter the name of the dataset: ")
                else:
                    dataset_name = input("Enter the name of the dataset: ")
            
            videos = get_playlist_videos(playlist_url)
            print(f"\nFound {len(videos)} videos in playlist")
            if videos:
                dataset_name = process_videos_automatic(model, videos, dataset_name)
            else:
                print("No videos found in playlist")
                continue

        elif choice == '3':
            file_path = input("Enter the path to the video links file: ")
            print("\nReading links and processing playlists...")
            videos = read_video_links_from_file(file_path)
            if not videos:
                print("No valid video links found in file")
                continue
            
            if last_dataset_name:
                print(f"\nLast used dataset name: {last_dataset_name}")
                use_last = input("Use this name? (y/n): ").lower().startswith('y')
                dataset_name = last_dataset_name if use_last else input("Enter the name of the dataset: ")
            else:
                dataset_name = input("Enter the name of the dataset: ")
                
            print(f"\nFound {len(videos)} total videos (including videos from playlists)")
            dataset_name = process_videos_automatic(model, videos, dataset_name)

        # Save the dataset name for future use
        if dataset_name:
            save_dataset_name(dataset_name)

if __name__ == "__main__":
    main()
