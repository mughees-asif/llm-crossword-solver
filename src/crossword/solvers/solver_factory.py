"""
Factory module to create appropriate crossword solvers based on puzzle characteristics.
"""

import os
import logging
from openai import AzureOpenAI
from typing import Optional

from ..crossword import CrosswordPuzzle
from .solver import CrosswordSolver
from .cryptic_solver import CrypticSolver


class SolverFactory:
    """
    Factory class to create appropriate solvers based on puzzle characteristics.
    """

    @staticmethod
    def create_solver(
        puzzle: CrosswordPuzzle,
        client: Optional[AzureOpenAI] = None,
        model: str = "gpt-4o",
        verbose: bool = False,
    ) -> CrosswordSolver:
        """
        Create an appropriate solver based on puzzle characteristics.

        Args:
            puzzle: The crossword puzzle to solve.
            client: Optional OpenAI client to use.
            model: The model to use for solving.
            verbose: Whether to print verbose output.

        Returns:
            An appropriate CrosswordSolver instance.
        """
        # If client is not provided, create one
        if client is None:
            client = AzureOpenAI(
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )

        # Check for cryptic puzzle based on clue text patterns
        is_cryptic = SolverFactory._is_cryptic_puzzle(puzzle)

        if is_cryptic:
            if verbose:
                logging.info("Creating CrypticSolver for cryptic puzzle")
            return CrypticSolver(client=client, model=model)
        else:
            if verbose:
                logging.info("Creating standard CrosswordSolver")
            return CrosswordSolver(client=client, model=model)

    @staticmethod
    def _is_cryptic_puzzle(puzzle: CrosswordPuzzle) -> bool:
        """
        Determine if a puzzle is cryptic based on clue text patterns.

        Args:
            puzzle: The crossword puzzle to analyze.

        Returns:
            True if the puzzle appears to be cryptic, False otherwise.
        """
        # Check clue patterns that are common in cryptic puzzles
        cryptic_word_indicators = [
            # Words that often indicate anagrams
            " arranged",
            " mixed",
            " confused",
            " disturbed",
            " muddled",
            " chaotic",
            " broken",
            " damaged",
            # Words that often indicate hidden answers
            " within",
            " hidden in",
            " concealed in",
            " part of",
            " held by",
            # Words that often indicate homophones
            " sounds like",
            " we hear",
            " they say",
            " audibly",
            # Words that often indicate reversals
            " back",
            " return",
            " reversed",
            " backwards",
            # Words that often indicate initial/final letters
            " heads",
            " initially",
            " first",
            " finally",
            " ultimately",
        ]

        # Common words that are NOT by themselves indicators of cryptic clues
        standard_clue_terms = [
            "(",
            ")",  # Parentheses used for length indication are common in standard puzzles
            " in ",
            " and ",
            " with ",
            " some ",
            " before ",
            " after ",
            " inside",  # These are common in regular clues too
        ]

        # Count the number of clues with cryptic indicators
        cryptic_count = 0
        total_clues = len(puzzle.clues)

        if total_clues == 0:
            return False

        for clue in puzzle.clues:
            clue_text = clue.text.lower()

            # Check for strong cryptic indicators
            if any(indicator in clue_text for indicator in cryptic_word_indicators):
                cryptic_count += 1
                continue

            # For weaker indicators (like parentheses), use a more complex check
            # In cryptic crosswords, parentheses often contain length AND wordplay hints
            if "(" in clue_text and ")" in clue_text:
                # Check if there's something before the parentheses that looks like wordplay
                before_paren = clue_text.split("(")[0].strip()

                # If the clue contains multiple words or specific cryptic patterns,
                # it's more likely to be cryptic
                word_count = len(before_paren.split())
                contains_cryptic_patterns = any(
                    term in before_paren
                    for term in ["?", "!", "perhaps", "maybe", "say"]
                )

                if word_count >= 3 or contains_cryptic_patterns:
                    cryptic_count += 1

        # If more than 25% of clues have cryptic indicators, consider it cryptic
        return cryptic_count / total_clues >= 0.25
