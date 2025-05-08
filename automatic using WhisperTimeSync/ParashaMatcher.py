import json
import os
from typing import Dict, Set, Tuple, Optional
from difflib import SequenceMatcher

class ParashaMatcher:
    def __init__(self, variants_file_path):
        self.variants_dict = self._load_variants(variants_file_path)
    
    def _load_variants(self, file_path) -> Dict[str, Set[str]]:
        """Load parasha variants from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Variants file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            variants_dict = json.load(f)
        
        # Convert lists to sets for faster lookup
        return {k: set(v) for k, v in variants_dict.items()}
    
    def match_parasha_name(self, input_name) -> Tuple[Optional[str], float, bool]:
        """Find matching parasha using exact or fuzzy matching.
        
        Returns:
        - matched_parasha: The matched parasha name or None if no match
        - confidence: A score between 0 and 1 indicating match confidence
        - is_exact: Boolean indicating if match was exact
        """
        if not input_name or not isinstance(input_name, str):
            return None, 0.0, False
        
        input_name = input_name.lower().strip()
        
        # Try exact matching first
        for parasha, variants in self.variants_dict.items():
            if input_name in variants or input_name == parasha.lower():
                return parasha, 1.0, True
        
        # If no exact match, try fuzzy matching
        best_match = None
        best_score = 0.0
        
        for parasha, variants in self.variants_dict.items():
            # Try matching with the main parasha name
            score = SequenceMatcher(None, input_name, parasha.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = parasha
            
            # Try matching with each variant
            for variant in variants:
                score = SequenceMatcher(None, input_name, variant).ratio()
                if score > best_score:
                    best_score = score
                    best_match = parasha
        
        # Return match only if confidence is high enough
        if best_score >= 0.7:
            return best_match, best_score, False
        
        return None, 0.0, False
