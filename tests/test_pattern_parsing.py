"""
Test script to demonstrate the pattern parsing enhancement for crossword solving.
This shows how the solver now understands word structure from clue patterns like "(7,3)".
"""

import json
import re
from typing import List, Optional

def parse_length_pattern(clue_text: str) -> Optional[List[int]]:
    """
    Extract length pattern from clue text.
    
    Examples:
    - "Greek tragedy (7,3)" -> [7, 3]
    - "Elliptical shape (4)" -> [4]
    - "Safety equipment for a biker, say (5,6)" -> [5, 6]
    
    Args:
        clue_text: The clue text to parse
        
    Returns:
        List of word lengths, or None if no pattern found
    """
    # Look for patterns like (7), (3,5), (5,6), (7,3) at end of clue
    pattern = r'\(([0-9,\s]+)\)$'
    match = re.search(pattern, clue_text.strip())
    
    if match:
        length_str = match.group(1)
        # Parse comma-separated lengths
        try:
            lengths = [int(x.strip()) for x in length_str.split(',')]
            return lengths
        except ValueError:
            return None
    
    return None


def demonstrate_pattern_parsing():
    """
    Demonstrate the pattern parsing functionality with examples from hard puzzle data.
    """
    print("=== Pattern Parsing Demonstration ===\n")
    
    # Test patterns from hard.json
    test_clues = [
        "Greek tragedy (7,3)",
        "A year (3,5)",
        "Elliptical shape (4)",
        "Safety equipment for a biker, say (5,6)",
        "Mess (4,6)",
        "Perform tricks (7)",
    ]
    
    print("Testing pattern parsing on hard puzzle clues:\n")
    
    for clue_text in test_clues:
        pattern = parse_length_pattern(clue_text)
        
        if pattern and len(pattern) > 1:
            print(f"✓ '{clue_text}'")
            print(f"  → Detected pattern: {pattern} (total: {sum(pattern)} letters)")
            print(
                f"  → Word structure: {len(pattern)} words of {'+'.join(map(str, pattern))} letters"
            )
            print(f"  → Example: If answer is 'OEDIPUSREX', it would be parsed as:")
            print(
                f"    - Word 1: 'OEDIPUS' ({pattern[0] if len(pattern) > 0 else 0} letters)"
            )
            print(
                f"    - Word 2: 'REX' ({pattern[1] if len(pattern) > 1 else 0} letters)"
            )
        else:
            print(f"• '{clue_text}'")
            print(
                f"  → Single word pattern: {pattern[0] if pattern else 'unknown'} letters"
            )
        print()


def demonstrate_enhanced_prompting():
    """
    Show how the enhanced prompting works with structure information.
    """
    print("=== Enhanced Prompting Demonstration ===\n")
    
    # Simulate what the LLM would receive with the enhanced prompting
    print("BEFORE Enhancement (old system):")
    print("1. Greek tragedy (10 letters, across)")
    print("2. A year (8 letters, across)")
    print("3. Safety equipment for a biker, say (11 letters, across)")
    print()
    
    print("AFTER Enhancement (new system):")
    print("1. Greek tragedy (This is 2 words: 7+3 letters)")
    print("2. A year (This is 2 words: 3+5 letters)")
    print("3. Safety equipment for a biker, say (This is 2 words: 5+6 letters)")
    print()
    
    print("Impact on LLM reasoning:")
    print("- OLD: LLM tries to find any 10-letter word for 'Greek tragedy'")
    print("- NEW: LLM knows to look for 7-letter word + 3-letter word")
    print("- OLD: Might suggest 'GREEKMYTH' or 'TRAGEDYONE'")
    print("- NEW: More likely to suggest 'OEDIPUSREX' (OEDIPUS + REX)")
    print()


def analyze_puzzle_patterns():
    """
    Analyze patterns in the actual puzzle data.
    """
    print("=== Puzzle Pattern Analysis ===\n")
    
    try:
        # Load and analyze hard puzzle
        with open("data/hard.json", "r") as f:
            hard_puzzle = json.load(f)
        
        # Load and analyze cryptic puzzle  
        with open("data/cryptic.json", "r") as f:
            cryptic_puzzle = json.load(f)
    except FileNotFoundError as e:
        print(f"Could not load puzzle data: {e}")
        return
    
    def analyze_puzzle_clues(puzzle_data, puzzle_name):
        print(f"{puzzle_name} Puzzle Analysis:")
        
        multi_word_count = 0
        single_word_count = 0
        
        for clue in puzzle_data["clues"]:
            pattern = parse_length_pattern(clue["text"])
            
            if pattern and len(pattern) > 1:
                multi_word_count += 1
                print(f"  Multi-word: {clue['text']} → {pattern}")
            else:
                single_word_count += 1
        
        total_clues = multi_word_count + single_word_count
        if total_clues > 0:
            print(f"  Summary: {multi_word_count} multi-word, {single_word_count} single-word clues")
            print(f"  Multi-word ratio: {multi_word_count/total_clues*100:.1f}%")
        print()
    
    analyze_puzzle_clues(hard_puzzle, "Hard")
    analyze_puzzle_clues(cryptic_puzzle, "Cryptic")


def show_expected_improvements():
    """
    Show the expected improvements from this enhancement.
    """
    print("=== Expected Improvements ===\n")
    
    improvements = [
        {
            "aspect": "Search Space Reduction",
            "before": "All possible 10-letter combinations",
            "after": "Only valid 7+3 letter word combinations",
            "impact": "Massive reduction in candidate space",
        },
        {
            "aspect": "LLM Accuracy",
            "before": "LLM guesses random 10-letter words",
            "after": "LLM understands word structure requirement",
            "impact": "Higher quality candidate generation",
        },
        {
            "aspect": "Constraint Satisfaction",
            "before": "Basic length-only constraints",
            "after": "Word boundary constraints",
            "impact": "Better intersection reasoning",
        },
        {
            "aspect": "Success Rate",
            "before": "Low success on hard/cryptic puzzles",
            "after": "Significantly improved success rate",
            "impact": "Solves previously unsolvable puzzles",
        },
    ]
    
    for improvement in improvements:
        print(f"📈 {improvement['aspect']}:")
        print(f"   BEFORE: {improvement['before']}")
        print(f"   AFTER:  {improvement['after']}")
        print(f"   IMPACT: {improvement['impact']}")
        print()


if __name__ == "__main__":
    demonstrate_pattern_parsing()
    demonstrate_enhanced_prompting()
    analyze_puzzle_patterns()
    show_expected_improvements()
    
    print("=== Test Complete ===")
    print("\nThe enhanced solver now:")
    print("✓ Parses word structure from clue patterns")
    print("✓ Provides structured hints to the LLM")
    print("✓ Validates answers against expected patterns")
    print("✓ Should significantly improve hard/cryptic puzzle success rates")
