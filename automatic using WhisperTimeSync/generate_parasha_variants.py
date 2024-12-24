import google.generativeai as genai
import json
import re
import time
from typing import Set, Dict, List, Optional
import atexit
import signal
import os

def clean_variant(name: str) -> str:
    """Remove all characters except lowercase English letters."""
    # Convert to lowercase first
    name = name.lower()
    # Keep only a-z characters
    return ''.join(c for c in name if c.isascii() and c.islower())

def clean_llm_response(response_text: str) -> List[str]:
    """Clean and parse LLM response to extract variants."""
    # Remove code block markers if present
    text = response_text.strip()
    if "```" in text:
        text = text[text.find("```"):text.rfind("```")]
        text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        # Try parsing as JSON array
        variants = json.loads(text)
        if isinstance(variants, list):
            return variants
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract array-like text
    text = text.strip("[]").strip()
    if text:
        # Split by comma and clean each item
        return [item.strip().strip('"\'').lower() for item in text.split(",")]
    
    return []

def get_variants_for_parasha(model, parasha: str, retry_count: int = 3) -> List[str]:
    """Generate spelling variants for a parasha name using LLM with retries."""
    prompt = f"""
    Generate ALL possible English spelling variants for the Hebrew Torah portion name "{parasha}".
    Include all common transliteration variations.

    Rules:
    1. Use only English letters (a-z)
    2. No spaces allowed
    3. All lowercase
    4. Include Ashkenazi and Sephardi pronunciation variants
    5. Include variants with and without doubled letters
    6. Include variants with different vowel representations

    Format: Return a simple comma-separated list.
    
    Example for "בראשית":
    ["bereshit", "bereishit", "bereshis", "bereishis", "breishit", "breishis", "bereyshit", "bereyshis"]

    For "{parasha}", provide ALL valid transliteration variants:
    """
    
    for attempt in range(retry_count):
        try:
            response = model.generate_content(prompt)
            # Increase wait time to avoid rate limiting
            time.sleep(10)  # Increased from 7 to 10 seconds
            
            variants = clean_llm_response(response.text)
            valid_variants = [
                clean_variant(variant) for variant in variants 
                if isinstance(variant, str) and clean_variant(variant)
            ]
            valid_variants.append(clean_variant(parasha))
            return valid_variants
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry_count} failed for {parasha}: {e}")
            if attempt < retry_count - 1:
                time.sleep(15)  # Wait longer between retries
            else:
                print(f"All attempts failed for {parasha}")
                return [clean_variant(parasha)]

def generate_all_variants(api_key: str, parsha_names: List[str]) -> Dict[str, str]:
    """Generate variants for all parasha names in reversed format (variant -> parasha)."""
    model = None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
            } 
        )
        
        variant_to_parasha = {}
        total = len(parsha_names)
        
        for idx, parasha in enumerate(parsha_names, 1):
            print(f"Processing {parasha} ({idx}/{total})...")
            cleaned_parasha = clean_variant(parasha)
            if not cleaned_parasha:
                print(f"Warning: Original parasha name '{parasha}' resulted in empty string after cleaning")
                continue
                
            # Add the canonical name as a variant
            variant_to_parasha[parasha.lower()] = parasha
            
            # Get and add all variants
            variants = get_variants_for_parasha(model, parasha)
            for variant in variants:
                if variant:  # Skip empty variants
                    variant_to_parasha[variant.lower()] = parasha
                        
            print(f"Generated variants for {parasha}: {variants}")
            if len(variants) < 3:
                print(f"Warning: Got only {len(variants)} variants for {parasha}")
            
            # Save progress after each parasha
            save_variants(variant_to_parasha, f"parasha_variants_progress_{idx}.json")
        
        # Clean up progress files after successful completion
        cleanup_progress_files()
        return variant_to_parasha
        
    except Exception as e:
        print(f"Fatal error in generate_all_variants: {e}")
        return variant_to_parasha
    finally:
        if model:
            try:
                # Explicitly close the model's session
                model._client.close()
                del model
            except:
                pass
        cleanup_resources()

def save_variants(variants_dict: Dict[str, str], filename: str = "parasha_variants.json"):
    """Save variants to a JSON file in the reversed format."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(variants_dict, f, indent=2, ensure_ascii=False, sort_keys=True)

def load_variants(filename: str = "parasha_variants.json") -> Dict[str, str]:
    """Load variants from a JSON file in the reversed format."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def cleanup_resources():
    """Cleanup function to handle GRPC resources."""
    try:
        # Force cleanup of any remaining GRPC channels
        import grpc
        for channel in list(grpc._channel._channel_pool):
            channel.close()
        # Clear the channel pool
        grpc._channel._channel_pool.clear()
    except:
        pass

def cleanup_progress_files():
    """Remove temporary progress files."""
    for filename in os.listdir('.'):
        if filename.startswith('parasha_variants_progress_') and filename.endswith('.json'):
            try:
                os.remove(filename)
                print(f"Removed progress file: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_resources)

# Add signal handlers
def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print("\nReceived termination signal. Cleaning up...")
    cleanup_resources()
    import sys
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    from get_data_from_youtube import get_api_key, parsha_names
    
    try:
        api_key = get_api_key()
        variants_dict = generate_all_variants(api_key, parsha_names[:2])
        save_variants(variants_dict)
        print("Successfully completed generating all variants!")
    except Exception as e:
        print(f"Program terminated with error: {e}")
    finally:
        cleanup_progress_files()  # Clean up progress files even if there's an error
        cleanup_resources()
        # Use os._exit instead of sys.exit to force immediate termination
        os._exit(0)
