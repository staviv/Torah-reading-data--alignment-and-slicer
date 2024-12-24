import google.generativeai as genai
import json
import re
import time
from typing import Set, Dict, List, Optional

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

def get_variants_for_parasha(model, parasha: str) -> Set[str]:
    """Generate spelling variants for a parasha name using LLM."""
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
    
    response = model.generate_content(prompt)
    time.sleep(7)  # Wait 7 seconds after getting LLM response
    try:
        variants = clean_llm_response(response.text)
        # Clean variants instead of filtering
        valid_variants = {
            clean_variant(variant) for variant in variants 
            if isinstance(variant, str) and clean_variant(variant)  # Only add non-empty results
        }
        valid_variants.add(clean_variant(parasha))  # Always include cleaned original name
        return valid_variants
        
    except Exception as e:
        print(f"Error generating variants for {parasha}: {e}")
        return {clean_variant(parasha)}

def generate_all_variants(api_key: str, parsha_names: List[str]) -> Dict[str, Set[str]]:
    """Generate variants for all parasha names."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={
            "temperature": 0,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
            } 
    )
    
    variants_dict = {}
    for parasha in parsha_names:
        cleaned_parasha = clean_variant(parasha)
        if not cleaned_parasha:
            print(f"Warning: Original parasha name '{parasha}' resulted in empty string after cleaning")
            continue
        variants = get_variants_for_parasha(model, parasha)
        variants_dict[parasha] = variants
        print(f"Generated variants for {parasha}: {variants}")
        if len(variants) < 3:
            print(f"Warning: Got only {len(variants)} variants for {parasha}")
    
    return variants_dict

def save_variants(variants_dict: Dict[str, Set[str]], filename: str = "parasha_variants.json"):
    """Save variants to a JSON file."""
    # Convert sets to lists for JSON serialization
    json_dict = {k: list(v) for k, v in variants_dict.items()}
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=2)

def load_variants(filename: str = "parasha_variants.json") -> Dict[str, Set[str]]:
    """Load variants from a JSON file."""
    try:
        with open(filename, 'r') as f:
            json_dict = json.load(f)
        return {k: set(v) for k, v in json_dict.items()}
    except FileNotFoundError:
        return {}

if __name__ == "__main__":
    from get_data_from_youtube import get_api_key, parsha_names
    
    api_key = get_api_key()
    variants_dict = generate_all_variants(api_key, parsha_names)
    save_variants(variants_dict)
