
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher
import json

class ParashaMatcher:
    def __init__(self, variants_file: str):
        # Load variants from file
        with open(variants_file, 'r') as f:
            parasha_to_variants = json.load(f)
        print(parasha_to_variants)
        # Create reverse mapping: variant -> canonical name
        self.variant_to_parasha: Dict[str, str] = {}
        for parasha, variants in parasha_to_variants.items():
            for variant in variants:
                self.variant_to_parasha[variant.lower()] = parasha
        
        # Store all variants for fuzzy matching
        self.all_variants = list(self.variant_to_parasha.keys())

    def find_exact_match(self, name: str) -> Optional[str]:
        """Find exact match in variants dictionary."""
        name = name.lower()
        return self.variant_to_parasha.get(''.join(c for c in name if c.isascii() and c.islower()))

    def find_closest_match(self, name: str, min_similarity: float = 0.8) -> Optional[Tuple[str, float]]:
        """Find the closest matching variant using sequence matching."""
        name = name.lower()
        best_ratio = 0
        best_match = None
        
        for variant in self.all_variants:
            ratio = SequenceMatcher(None, name, variant).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = variant
        
        if best_ratio >= min_similarity and best_match:
            return (self.variant_to_parasha[best_match], best_ratio)
        return None

    def match_parasha_name(self, name: str, min_similarity: float = 0.8) -> Tuple[Optional[str], float, bool]:
        """
        Try to match a parasha name using exact and fuzzy matching.
        Returns: (matched_parasha, confidence, is_exact_match)
        """
        # Try exact match first
        exact_match = self.find_exact_match(name)
        if exact_match:
            return (exact_match, 1.0, True)
        
        # Try fuzzy match
        fuzzy_match = self.find_closest_match(name, min_similarity)
        if fuzzy_match:
            return (fuzzy_match[0], fuzzy_match[1], False)
        
        return (None, 0.0, False)