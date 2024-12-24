import yt_dlp
import google.generativeai as genai
import json
import os
import sys
from typing import Dict, Set, Optional
from get_all_aliyot_from_sefaria import parsha_names # parsha_names is a list of all the parshiot in the Torah
from generate_parasha_variants import load_variants, clean_variant


def get_api_key():
    # Load API key from a file
    try:
        with open("gemini_token", "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        api_key = ""
        
    if api_key == "":
        # Prompt user for API key if not found in environment
        api_key = input("Please enter your Google API key: ").strip()
        if not api_key:
            print("Error: No API key provided. Please set GOOGLE_API_KEY environment variable or enter key when prompted.")
            sys.exit(1)
        
        # Save the token to a file for future use
        with open("gemini_token", "w") as f:
            f.write(api_key)
    
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
        Set is_parasha to TRUE if you can understand which parasha is being read in the video.
        
        in parasha FIELD you must return exactly one of
        the valid parasha names from this list:
        {{parsha_names}}
        
        The response must include the following fields:
        
        {{
            "is_parasha": boolean (true ONLY if all conditions above are met),
            "parasha": string (must be exactly one of the valid parasha names, or null if not a weekly torah reading),
            "aliyah": string (either "all" for full parasha or "1"-"7" for specific aliyah),
            "dataset_name": string (suggested name for the dataset, or null if not applicable),
            "confidence": float (0-1, how confident you are in the prediction)
        }}

        Example outputs:
        {{"is_parasha": true, "parasha": "Bereshit", "aliyah": "1", "dataset_name": "Nusach-Ashkenaz-david-goldberg", "confidence": 0.95}}
        {{"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": null, "confidence": 0.9}} <- Not a Torah reading
        {{"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": null, "confidence": 0.8}} <- Torah lesson but not reading
        {{"is_parasha": false, "parasha": "Bereshit", "aliyah": null, "dataset_name": null, "confidence": 0.7}} <- Mentions parasha but not reading
        """
        
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
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

def get_llm_suggestion(model, link):
    metadata = get_video_metadata(link)
    prompt = f"""
    Analyze this YouTube video metadata to suggest parameters:
    Title: {metadata.get('title', '')}
    Description: {metadata.get('description', '')} 
    Channel: {metadata.get('channel', '')}
    """
    response = model.generate_content(prompt)
    result = clean_llm_json_response(response.text)
    if not result:
        return {}
    
    # Print metadata and confidence for user reference
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

def download_audio(link, output_path, start_time=None, end_time=None):
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"\nWarning: File already exists at {output_path}")
        if not input("Do you want to overwrite it? (y/n): ").lower().startswith('y'):
            print("Skipping download...")
            return False

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': f'{output_path}',
    }

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

def find_matching_parasha(name: str, variants_dict: Dict[str, Set[str]]) -> Optional[str]:
    """Find matching parasha using pre-generated variants."""
    if not name or not isinstance(name, str):  # Check if name is a string and not empty
        return None
        
    name = clean_variant(name)
    if not name:  # If name is empty after cleaning
        return None
    
    print(f"\nTrying to match '{name}' to known parasha variants...")
    for parasha, variants in variants_dict.items():
        print(f"Checking {parasha}...")
        if name in variants:
            return parasha
    return None

def match_parasha_name(model, suggested_name):
    # Load pre-generated variants
    variants_dict = load_variants("automatic using WhisperTimeSync/parasha_variants.json")
    
    if (not suggested_name) or (suggested_name in parsha_names):
        return suggested_name
        
    # Try to match using variants
    matched_name = find_matching_parasha(suggested_name, variants_dict)
    if matched_name:
        print(f"\nMatched '{suggested_name}' to '{matched_name}' using variant matching")
        return matched_name
    
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
    if suggested_parasha not in parsha_names:
        print(f"\nSuggested parasha name: {suggested_parasha}")
        print("\nThis parasha name is not in the valid list")
        print("\nTrying to match suggested parasha name to valid list...")
        matched_parasha = match_parasha_name(model, suggested_parasha)
        if matched_parasha:
            suggestion["parasha"] = matched_parasha
        else:
            print("\nWarning: Could not match parasha name to valid list")
            print(matched_parasha)
            if not input("Do you want to continue processing this current video (otherwise it will be skipped)? (y/n): ").lower().startswith('y'):
                return False, dataset_name

    print_parshiot()
    
    if suggestion.get("parasha") in parsha_names:
        suggested_parsha_num = parsha_names.index(suggestion.get("parasha")) + 1
    else:
        suggested_parsha_num = ""

    parsha_num = int(get_input_with_suggestion(
        f"Enter the number of the parsha {parsha_names[suggested_parsha_num - 1]} ", 
        suggested_parsha_num, 
        lambda x: x.isdigit() and 1 <= int(x) <= len(parsha_names)))

    parsha = parsha_names[parsha_num - 1]
    
    aliyah = get_input_with_suggestion(
        "Enter aliyah number (1-7) or 'all' for full parasha: ", 
        suggestion.get("aliyah"), 
        lambda x: x == "all" or (x.isdigit() and 1 <= int(x) <= 7))

    if dataset_name is None:
        last_dataset_name = get_last_dataset_name()
        if last_dataset_name:
            use_last = input(f"Use last dataset name '{last_dataset_name}'? (y/n): ").lower() == 'y'
            dataset_name = last_dataset_name if use_last else input("Enter the name of the dataset: ")
        else:
            dataset_name = input("Enter the name of the dataset: ")

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

def main():
    api_key = get_api_key()
    model = setup_llm(api_key)
    
    last_dataset_name = get_last_dataset_name()
    dataset_name = None
    
    while True:
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
        
        # Save the dataset name for future use
        if dataset_name:
            save_dataset_name(dataset_name)

if __name__ == "__main__":
    main()
