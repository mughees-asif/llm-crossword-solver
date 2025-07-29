"""
Specialized solver for cryptic crossword clues.
"""

import re
from typing import List, Dict, Optional
import logging

from .solver import CrosswordSolver
from ..crossword import CrosswordPuzzle
from ...utils.types import Clue

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CrypticSolver(CrosswordSolver):
    """
    A specialized solver for cryptic crossword puzzles.

    This extends the base CrosswordSolver with additional strategies for solving cryptic clues.
    """

    def __init__(self, **kwargs):
        """Initialize with parent class constructor."""
        super().__init__(**kwargs)

    def solve(self, puzzle: CrosswordPuzzle, verbose: bool = False) -> CrosswordPuzzle:
        """
        Override solve method to apply cryptic-specific strategies.
        """
        if verbose:
            logging.info("Using specialized cryptic solver")

        # Use the base solver but with enhanced prompting for cryptic clues
        return super().solve(puzzle, verbose)

    def _generate_batch_candidates(
        self, clues: List[Clue], difficulty: str
    ) -> Dict[int, List[str]]:
        """
        Generate candidate answers for cryptic clues with specialized prompting.

        Args:
            clues: The batch of clues to process.
            difficulty: The determined difficulty.

        Returns:
            Dictionary mapping batch indexes to lists of candidate answers.
        """
        # For cryptic clues, always use the cryptic prompt regardless of the detected difficulty
        system_prompt = """
        You are an expert cryptic crossword solver with decades of experience. Cryptic clues contain wordplay and often have misleading surface readings.
        
        Common types of cryptic clues include:
        1. Double definition: Two different meanings lead to the same word.
            Example: "Fast food (4)" - DIET (fast = abstain from food, food = diet)
        
        2. Anagrams: Words are rearranged, indicated by words like "broken", "confused", "disordered", "arranged", "sorted", etc.
            Example: "Scared rascal dances (9)" - SCATTERS (anagram of "scared rascal")
        
        3. Hidden words: The answer is hidden directly in the clue, indicated by words like "in", "within", "part of", "hiding in".
            Example: "Caught in sea, treasure (4)" - EASE (hidden in "sea treasure")
        
        4. Charades: The answer is formed by joining smaller words together.
            Example: "Fruit cake (5)" - FRUITCAKE (fruit + cake)
        
        5. Container: One word inside another, indicated by "in", "within", "around", "holding", "containing".
            Example: "Church officer keeping Eliot's cat (6)" - VESTRY (contains "EST" from "Eliot's" within "VRY" (very without the 'e'))
        
        6. Reversal: Words spelled backward, indicated by "back", "returning", "reversed", etc.
            Example: "Send back Oriental drink (3)" - TEA (AET reversed)
        
        7. Initial/final letters: Taking first or last letters, indicated by "initially", "finally", "heads", "tails", etc.
            Example: "Leaders of major accounting firms enjoy riches (5)" - MAFER (first letters)
        
        8. Homophone: Words that sound alike, indicated by "sounds like", "they say", "heard", etc.
            Example: "We hear it's a judicial unit (5)" - COURT (sounds like "caught")
        
        Carefully analyze each clue to identify the wordplay techniques being used. Separate the definition part from the wordplay part.
        For each clue, provide your best answer along with a detailed explanation of how you solved it.
        """

        # Prepare the user prompt with the clues and detailed instructions for cryptic solving
        user_prompt = """Please solve these cryptic crossword clues:

"""
        for idx, clue in enumerate(clues):
            user_prompt += (
                f"{idx+1}. {clue.text} ({clue.length} letters, {clue.direction})\n"
            )

        user_prompt += """
For each clue, provide:
1. Your best answer (and alternatives if you have them)
2. A brief explanation of how you solved it, identifying:
   - The definition part of the clue
   - The wordplay part of the clue
   - The specific cryptic technique(s) used

Format your response as a JSON object with clue numbers as keys and arrays of possible answers as values. 
List the most likely answer first in each array.

Example format:
{
  "1": ["ANSWER1", "ALTERNATIVE1", "ALTERNATIVE2"],
  "2": ["ANSWER2", "ALTERNATIVE1"]
}
"""

        # Use parent class method but with our specialized prompts
        return self._call_llm_for_cryptic(clues, system_prompt, user_prompt)

    def _call_llm_for_cryptic(
        self, clues: List[Clue], system_prompt: str, user_prompt: str
    ) -> Dict[int, List[str]]:
        """Call LLM with cryptic-specific handling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic answers
            )

            # Extract and parse the response
            response_text = response.choices[0].message.content

            # Handle the response
            try:
                import json

                answers_dict = json.loads(response_text)

                # Convert to our expected format
                result = {}
                for idx in range(len(clues)):
                    key = str(
                        idx + 1
                    )  # The keys in the response should be "1", "2", etc.
                    if key in answers_dict:
                        # Get the answers, ensure they're uppercase and the right length
                        answers = answers_dict[key]
                        if isinstance(answers, list):
                            filtered_answers = [
                                answer.upper()
                                for answer in answers
                                if isinstance(answer, str)
                                and len(answer.strip()) == clues[idx].length
                            ]
                            result[idx] = filtered_answers
                        elif isinstance(answers, str):
                            # Handle case where LLM returns a single string instead of a list
                            if len(answers.strip()) == clues[idx].length:
                                result[idx] = [answers.upper()]

                return result
            except Exception as e:
                logging.error(f"Error parsing LLM response: {e}")
                logging.error(f"Response was: {response_text}")
                return {idx: [] for idx in range(len(clues))}

        except Exception as e:
            logging.error(f"API call failed: {e}")
            return {idx: [] for idx in range(len(clues))}

    def _analyze_cryptic_clue(self, clue_text: str) -> Dict:
        """
        Analyze a cryptic clue to identify its components.

        Args:
            clue_text: The text of the cryptic clue.

        Returns:
            Dictionary with analysis of the clue components.
        """
        # This is a placeholder for more sophisticated cryptic clue analysis
        # In a real implementation, we would parse the clue to identify:
        # - Definition part
        # - Wordplay part
        # - Type of wordplay (anagram, hidden word, etc.)

        # Example analysis structure
        analysis = {
            "full_clue": clue_text,
            "definition_part": None,
            "wordplay_part": None,
            "wordplay_types": [],
            "indicators": [],
        }

        # Check for common cryptic indicators
        indicators = {
            "anagram": [
                "arranged",
                "broken",
                "confused",
                "damaged",
                "odd",
                "out",
                "scattered",
                "scrambled",
                "mixed",
                "troubled",
                "irregular",
                "crazy",
                "wild",
                "muddled",
            ],
            "hidden": [
                "hidden",
                "in",
                "inside",
                "within",
                "hiding",
                "concealing",
                "some",
                "part of",
                "held by",
                "housed by",
            ],
            "reversal": [
                "back",
                "return",
                "reversed",
                "going west",
                "going up",
                "ascending",
                "rising",
            ],
            "initial": ["first", "head", "start", "beginning", "initially"],
            "final": ["last", "end", "finally", "ultimately", "tail"],
            "homophone": [
                "sounds like",
                "heard",
                "they say",
                "audibly",
                "we hear",
                "reportedly",
            ],
        }

        for technique, inds in indicators.items():
            for indicator in inds:
                if re.search(r"\b" + indicator + r"\b", clue_text.lower()):
                    analysis["indicators"].append(indicator)
                    analysis["wordplay_types"].append(technique)

        return analysis
