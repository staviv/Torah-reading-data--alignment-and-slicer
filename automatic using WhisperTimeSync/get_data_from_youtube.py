import yt_dlp
import google.generativeai as genai
import json
import os
import sys

def get_api_key():
    # Try environment variable first
    api_key = "AIzaSyA9eKU-M8RGCf7yZEcaBL5XZZWejhDl9tQ"
    
    if not api_key:
        # Prompt user for API key if not found in environment
        api_key = input("Please enter your Google API key: ").strip()
        if not api_key:
            print("Error: No API key provided. Please set GOOGLE_API_KEY environment variable or enter key when prompted.")
            sys.exit(1)
        
        # Optionally save to environment for current session
        os.environ["GOOGLE_API_KEY"] = api_key
    
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
        You are a Torah reading assistant. When given a YouTube URL and its metadata, analyze it and return a JSON object with the following structure:
        {
            "parasha": string (must be exactly one of the valid parasha names),
            "aliyah": integer (1-7),
            "dataset_name": string (suggested name for the dataset),
            "confidence": float (0-1, how confident you are in the prediction)
        }
        You must use the following parasha names:
        parsha_names = ["Bereshit", "Noach", "LechLecha", "Vayera", "ChayeiSara", "Toldot", "Vayetzei", "Vayishlach", "Vayeshev", "Miketz", "Vayigash", "Vayechi", "Shemot", "Vaera", "Bo", "Beshalach", "Yitro", "Mishpatim", "Terumah", "Tetzaveh", "KiTisa", "Vayakhel", "Pekudei", "Vayikra", "Tzav", "Shmini", "Tazria", "Metzora", "AchreiMot", "Kedoshim", "Emor", "Behar", "Bechukotai", "Bamidbar", "Nasso", "Behaalotcha", "Shlach", "Korach", "Chukat", "Balak", "Pinchas", "Matot", "Masei", "Devarim", "Vaethanan", "Eikev", "Reeh", "Shoftim", "KiTeitzei", "KiTavo", "Nitzavim", "Vayeilech", "Haazinu", "VezotHaberakhah"]

        Use the video title, description, and other metadata to make more accurate predictions.
        If the content seems unrelated to Torah reading, set confidence to 0.

        Example outputs:
        {"parasha": "Bereshit", "aliyah": 1, "dataset_name": "NusachAshkenaz-david-goldberg", "confidence": 0.95}
        {"parasha": "Vayera", "aliyah": 3, "dataset_name": "NusachSefard-isaac-levy", "confidence": 0.85}
        """
        
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config=generation_config,
            system_instruction=system_instruction
        )
    except Exception as e:
        print(f"Error setting up LLM: {str(e)}")
        sys.exit(1)

def get_llm_suggestion(model, link):
    metadata = get_video_metadata(link)
    prompt = f"""
    Analyze this YouTube video and its metadata to suggest parameters:
    URL: {link}
    Title: {metadata.get('title', '')}
    Description: {metadata.get('description', '')} 
    Channel: {metadata.get('channel', '')}
    Duration: {metadata.get('duration', 0)} seconds
    """
    response = model.generate_content(prompt)
    # Remove code block markers if present
    # Extract JSON content between ```json and ``` markers
    start = response.text.find("```json") + 7  # Skip past ```json
    end = response.text.find("```", start)
    cleaned_text = response.text[start:end].strip() if start >= 7 and end != -1 else response.text.strip()
    result = json.loads(cleaned_text)
    
    # Print metadata and confidence for user reference
    print("\nVideo metadata:")
    print(f"Title: {metadata.get('title', 'N/A')}")
    print(f"Channel: {metadata.get('channel', 'N/A')}")
    print(f"Duration: {metadata.get('duration', 0)} seconds")
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


def print_parshiot():
    # Show aliyah range for the selected parsha and get confirmation
    print("\nHere are all parshiot with their numbers:")
    for i, p in enumerate(parsha_names, 1):
        print(f"{p}: {i}", end="  ")
        if i % 4 == 0:  # Print 4 parshiot per line
            print()

def main():
    api_key = get_api_key()
    model = setup_llm(api_key)
    
    link = input("Enter the link of the youtube video: ")
    suggestion = get_llm_suggestion(model, link)
    
    from get_all_aliyot_from_sefaria import parsha_names # parsha_names is a list of all the parshiot in the Torah
    # Show available parshiot and let user choose
    print_parshiot()
        
    if suggestion.get("parasha") in parsha_names:
        suggested_parsha_num = parsha_names.index(suggestion.get("parasha")) + 1
    else:
        suggested_parsha_num = ""

    parsha_num = int(get_input_with_suggestion(
        f"Enter the number of the parsha {parsha_names[suggested_parsha_num - 1]} ", suggested_parsha_num, lambda x: x.isdigit() and 1 <= int(x) <= len(parsha_names)))

    parsha = parsha_names[parsha_num - 1]

    # Let user choose aliyah number
    aliyah = int(get_input_with_suggestion(
        "Enter aliyah number (1-7): ", suggestion.get("aliyah"), lambda x: x.isdigit() and 1 <= int(x) <= 7))

    print(f"\nSelected: Parshat {parsha}, Aliyah {aliyah}")
    
    print()

    # Get dataset name
    dataset_name = get_input_with_suggestion(
        "Enter the name of the dataset: ", suggestion.get("dataset_name"))
    dataset_dir = f"/home/prj8045/data/{dataset_name}"

    confirm = input(f"\nConfirm selection (Parsha {parsha}:{parsha_num}, Aliyah {aliyah}, Dataset {dataset_name})? (y/n): ")
    while confirm.lower() not in ['y', 'n']:
        confirm = input("Please enter y or n: ")

    if confirm.lower() == 'y':
        print("\nProceeding with download and processing...")
        audio_path = f"{dataset_dir}/{parsha}-{aliyah}.mp3"
        download_audio(link, audio_path)
        print(f"\nAudio downloaded to {audio_path}")
    else:
        print("\nCancelled. Please run again with different selections.")

if __name__ == "__main__":
    main()
