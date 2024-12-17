import yt_dlp
import google.generativeai as genai
import json
import os
import sys
from get_all_aliyot_from_sefaria import parsha_names # parsha_names is a list of all the parshiot in the Torah



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
                'upload_date': info.get('upload_date', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
            }
        except Exception as e:
            print(f"Warning: Could not extract video metadata: {str(e)}")
            return {}

def is_playlist(url):
    return 'playlist' in url or '?list=' in url

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
        }
        
        system_instruction = """
        You are a Torah reading assistant. When given a YouTube video's metadata, analyze it and return a JSON object.
        
        IMPORTANT RULES FOR is_parasha FIELD:
        Set is_parasha to TRUE if and only if ALL these conditions are met:
        1. The video contains an actual Torah reading (Kriat HaTorah)
        2. The reading is specifically from one of the weekly Torah portions (Parshiot Hashavua)
        3. You can clearly identify which specific parasha is being read
        
        Set is_parasha to FALSE in any of these cases:
        - If it's not a Torah reading at all
        - If it's a Torah reading but not from weekly portions (e.g., holidays, Rosh Chodesh)
        - If you can't clearly identify which specific parasha is being read
        - If you're unsure or the content is ambiguous
        - If it's a lesson about the parasha but not the actual reading
        - If it's just someone discussing or teaching the parasha
        
        The response must include the following fields:
        
        {
            "is_parasha": boolean (true ONLY if all conditions above are met),
            "parasha": string (must be exactly one of the valid parasha names, or null if not a weekly torah reading),
            "aliyah": string (either "all" for full parasha or "1"-"7" for specific aliyah),
            "dataset_name": string (suggested name for the dataset, or null if not applicable),
            "confidence": float (0-1, how confident you are in the prediction)
        }

        Example outputs:
        {"is_parasha": true, "parasha": "Bereshit", "aliyah": "1", "dataset_name": "Nusach-Ashkenaz-david-goldberg", "confidence": 0.95}
        {"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": null, "confidence": 0.9} <- Not a Torah reading
        {"is_parasha": false, "parasha": null, "aliyah": null, "dataset_name": null, "confidence": 0.8} <- Torah lesson but not reading
        {"is_parasha": false, "parasha": "Bereshit", "aliyah": null, "dataset_name": null, "confidence": 0.7} <- Mentions parasha but not reading
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

def download_audio(link, output_path):
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

def match_parasha_name(model, suggested_name):
    if not suggested_name or suggested_name in parsha_names:
        return suggested_name
        
    prompt = f"""
    Given this parasha name: "{suggested_name}"
    Match it to one of the valid parasha names from this list:
    {parsha_names}
    
    Return a JSON response in this format:
    {{
        "matched_parasha": "name from the list or null if no match found",
        "confidence": float between 0-1,
        "reasoning": "brief explanation of the match or why no match was found"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        result = clean_llm_json_response(response.text)
        if not result:
            return None
        
        matched_name = result.get("matched_parasha")
        confidence = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")
        
        if matched_name in parsha_names and confidence > 0.7:
            print(f"\nMatched '{suggested_name}' to '{matched_name}'")
            print(f"Reasoning: {reasoning}")
            print(f"Confidence: {confidence}")
            return matched_name
            
        return None
        
    except Exception as e:
        print(f"Error matching parasha name: {e}")
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
        print("\nTrying to match suggested parasha name to valid list...")
        matched_parasha = match_parasha_name(model, suggested_parasha)
        if matched_parasha:
            suggestion["parasha"] = matched_parasha
        else:
            print("\nWarning: Could not match parasha name to valid list")
            if not input("Do you want to continue anyway? (y/n): ").lower().startswith('y'):
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
    
    print(f"\nDownloading to {audio_path}...")
    if not download_audio(link, audio_path):
        return False, dataset_name
    print("Download complete")
    
    return True, dataset_name

def main():
    api_key = get_api_key()
    model = setup_llm(api_key)
    
    last_dataset_name = get_last_dataset_name()
    dataset_name = None
    
    while True:
        link = input("\nEnter the YouTube video/playlist link (or 'q' to quit): ")
        if link.lower() == 'q':
            break
            
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
