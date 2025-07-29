"""
Crossword Solver module that uses LLMs to solve crossword puzzles.
"""

import os
from openai import AzureOpenAI
from typing import List, Dict, Optional
import logging
import time
import re

from ..crossword import CrosswordPuzzle
from ...utils.types import Clue

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CrosswordSolver:
    """
    A class that uses LLM to solve crossword puzzles.
    """

    def __init__(
        self,
        client: AzureOpenAI = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the CrosswordSolver.

        Args:
            client: The OpenAI client to use.
            model: The model to use for solving.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        if client is None:
            self.client = AzureOpenAI(
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
        else:
            self.client = client

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def parse_length_pattern(self, clue_text: str) -> Optional[List[int]]:
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
        pattern = r"\(([0-9,\s]+)\)$"
        match = re.search(pattern, clue_text.strip())

        if match:
            length_str = match.group(1)
            # Parse comma-separated lengths
            try:
                lengths = [int(x.strip()) for x in length_str.split(",")]
                return lengths
            except ValueError:
                return None

        return None

    def validate_word_structure(self, answer: str, length_pattern: List[int]) -> bool:
        """
        Validate that answer matches expected word structure.

        Args:
            answer: The candidate answer
            length_pattern: Expected word lengths

        Returns:
            True if answer matches the pattern
        """
        if sum(length_pattern) != len(answer):
            return False

        # For basic validation, just check the total length matches
        # In a more sophisticated version, we could check word boundaries
        return True

    def solve(self, puzzle: CrosswordPuzzle, verbose: bool = False) -> CrosswordPuzzle:
        """
        Solve the given crossword puzzle.

        Args:
            puzzle: The crossword puzzle to solve.
            verbose: Whether to print verbose output.

        Returns:
            The solved puzzle.
        """
        # First, reset the puzzle to make sure we start fresh
        puzzle.reset()

        if verbose:
            logging.info(f"Solving puzzle with {len(puzzle.clues)} clues")
            logging.info(f"Grid size: {puzzle.width}x{puzzle.height}")

        # Always use LLM to solve the puzzle even if answers are provided in the data
        if verbose:
            logging.info("Using LLM to generate answers for all clues.")
        return self._solve_with_llm(puzzle, verbose)

    def _solve_with_answers(
        self, puzzle: CrosswordPuzzle, verbose: bool
    ) -> CrosswordPuzzle:
        """
        Solve the puzzle using provided answers.

        Args:
            puzzle: The crossword puzzle with answers.
            verbose: Whether to print verbose output.

        Returns:
            The solved puzzle.
        """
        # Simply reveal all answers
        puzzle.reveal_all()
        return puzzle

    def _solve_with_llm(
        self, puzzle: CrosswordPuzzle, verbose: bool
    ) -> CrosswordPuzzle:
        """
        Solve the puzzle using LLM to generate answers.

        Args:
            puzzle: The crossword puzzle to solve.
            verbose: Whether to print verbose output.

        Returns:
            The solved puzzle.
        """
        # 1. First, identify puzzle difficulty to adjust our strategy
        difficulty = self._determine_difficulty(puzzle)
        if verbose:
            logging.info(f"Determined puzzle difficulty: {difficulty}")

        # 2. Generate candidate answers for each clue
        candidates = self._generate_candidate_answers(puzzle, difficulty, verbose)
        if verbose:
            logging.info(
                f"Generated {sum(len(cands) for cands in candidates.values())} candidate answers"
            )

        # 3. Solve the crossword using constraint satisfaction
        solved_puzzle = self._solve_with_constraints(puzzle, candidates, verbose)

        return solved_puzzle

    def _determine_difficulty(self, puzzle: CrosswordPuzzle) -> str:
        """
        Determine the difficulty of the puzzle based on size and clue types.

        Args:
            puzzle: The crossword puzzle.

        Returns:
            String indicating difficulty: 'easy', 'medium', 'hard', or 'cryptic'
        """
        # Check for cryptic clues by looking for certain patterns in the clues
        cryptic_indicators = ["(", ")", "?"]
        has_cryptic_clues = any(
            any(ind in clue.text for ind in cryptic_indicators) for clue in puzzle.clues
        )

        # Base difficulty on grid size and presence of cryptic clues
        if puzzle.width <= 5 and puzzle.height <= 5:
            return "easy"
        elif puzzle.width <= 8 and puzzle.height <= 8:
            return "medium"
        elif has_cryptic_clues:
            return "cryptic"
        else:
            return "hard"

    def _generate_candidate_answers(
        self, puzzle: CrosswordPuzzle, difficulty: str, verbose: bool
    ) -> Dict[int, List[str]]:
        """
        Generate candidate answers for each clue using LLM.

        Args:
            puzzle: The crossword puzzle.
            difficulty: The determined difficulty.
            verbose: Whether to print verbose output.

        Returns:
            Dictionary mapping clue indexes to lists of candidate answers.
        """
        candidates = {}

        # Process clues in batches to reduce API calls
        batch_size = 5  # Process 5 clues at a time
        for i in range(0, len(puzzle.clues), batch_size):
            batch_clues = puzzle.clues[i : i + batch_size]

            if verbose:
                logging.info(
                    f"Processing batch of {len(batch_clues)} clues ({i+1} to {min(i+batch_size, len(puzzle.clues))})"
                )

            # Generate candidates for this batch
            batch_candidates = self._generate_batch_candidates(batch_clues, difficulty)

            # Store the candidates
            for idx, clue in enumerate(batch_clues):
                clue_index = puzzle.clues.index(clue)
                candidates[clue_index] = batch_candidates.get(idx, [])

                if verbose and candidates[clue_index]:
                    logging.info(
                        f"Clue {clue_index+1}: '{clue.text}' - Top candidates: {candidates[clue_index][:3]}"
                    )

        return candidates

    def _generate_batch_candidates(
        self, clues: List[Clue], difficulty: str
    ) -> Dict[int, List[str]]:
        """
        Generate candidate answers for a batch of clues using LLM.

        Args:
            clues: The batch of clues to process.
            difficulty: The determined difficulty.

        Returns:
            Dictionary mapping batch indexes to lists of candidate answers.
        """
        # Prepare the prompt based on difficulty
        if difficulty == "cryptic":
            system_prompt = """
            You are an expert crossword solver specializing in cryptic clues. Cryptic clues often contain wordplay and misleading definitions.
            For each clue, provide the most likely answer and explain your reasoning.
            Here are some common types of cryptic clues:
            1. Double definition: Two different meanings lead to the same word.
            2. Anagrams: Indicated by words like "broken", "confused", "disordered".
            3. Hidden words: The answer is hidden within the clue, indicated by words like "in", "within", "part of".
            4. Charades: The answer is broken into parts, each clued separately.
            5. Container: One word is placed inside another, indicated by "in", "within", "around".
            """
        elif difficulty == "hard":
            system_prompt = """
            You are an expert crossword solver with extensive knowledge of vocabulary, trivia, and wordplay.
            For each clue, provide the most likely answer and explain your reasoning briefly.
            """
        else:
            system_prompt = """
            You are helping to solve a crossword puzzle.
            For each clue, provide the most likely answer that fits the length requirement.
            """

        # Prepare the user prompt with enhanced structure information
        user_prompt = "Please solve the following crossword clues:\n\n"
        for idx, clue in enumerate(clues):
            # Parse length pattern for enhanced prompting
            length_pattern = self.parse_length_pattern(clue.text)

            if length_pattern and len(length_pattern) > 1:
                structure_hint = (
                    f" (This is {len(length_pattern)} words: "
                    f"{'+'.join(map(str, length_pattern))} letters)"
                )
            else:
                structure_hint = f" ({clue.length} letters)"

            user_prompt += f"{idx+1}. {clue.text}{structure_hint}\n"

        user_prompt += (
            "\nFor each clue, provide the best answer and any "
            "alternatives. For multi-word answers, provide them "
            "as single words without spaces (e.g., 'OEDIPUSREX' "
            "not 'OEDIPUS REX'). Format your response as a JSON "
            "object with clue numbers as keys and arrays of "
            "possible answers as values. List the most likely "
            "answer first."
        )

        # Make the API call with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
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
                    # Continue with retry

            except Exception as e:
                logging.error(f"API call attempt {attempt+1} failed: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

        # If all retries failed, return empty results
        return {idx: [] for idx in range(len(clues))}

    def _solve_with_constraints(
        self, puzzle: CrosswordPuzzle, candidates: Dict[int, List[str]], verbose: bool
    ) -> CrosswordPuzzle:
        """
        Solve the crossword using constraint satisfaction with the candidate answers.

        Args:
            puzzle: The crossword puzzle.
            candidates: Dictionary of candidate answers for each clue.
            verbose: Whether to print verbose output.

        Returns:
            The solved puzzle.
        """
        # Start by sorting clues by number of candidates (most constrained first)
        sorted_clue_indices = sorted(
            candidates.keys(), key=lambda idx: len(candidates[idx])
        )

        # Try to solve with backtracking
        success = self._backtrack(puzzle, sorted_clue_indices, candidates, 0, verbose)

        if not success and verbose:
            logging.warning("Backtracking failed to find a complete solution.")

        return puzzle

    def _backtrack(
        self,
        puzzle: CrosswordPuzzle,
        sorted_clue_indices: List[int],
        candidates: Dict[int, List[str]],
        index: int,
        verbose: bool,
    ) -> bool:
        """
        Backtracking algorithm to solve the puzzle with constraints.

        Args:
            puzzle: The crossword puzzle.
            sorted_clue_indices: List of clue indices sorted by constraint.
            candidates: Dictionary of candidate answers for each clue.
            index: Current index in the sorted_clue_indices list.
            verbose: Whether to print verbose output.

        Returns:
            True if a valid solution was found, False otherwise.
        """
        # Base case: if we've processed all clues, check if the puzzle is valid
        if index >= len(sorted_clue_indices):
            return puzzle.validate_all()

        # Get the current clue index and the corresponding clue
        clue_idx = sorted_clue_indices[index]
        clue = puzzle.clues[clue_idx]

        # Try each candidate answer
        for candidate in candidates.get(clue_idx, []):
            # Check if this candidate is compatible with existing filled cells
            if self._is_compatible(puzzle, clue, candidate):
                # Try this candidate
                if verbose:
                    logging.info(
                        f"Trying '{candidate}' for clue {clue_idx+1}: '{clue.text}'"
                    )

                puzzle.set_clue_chars(clue, list(candidate))

                # Recursively solve the rest
                if self._backtrack(
                    puzzle, sorted_clue_indices, candidates, index + 1, verbose
                ):
                    return True

                # If we get here, this candidate didn't work, so undo
                puzzle.undo()

        # No candidates worked
        return False

    def _is_compatible(
        self, puzzle: CrosswordPuzzle, clue: Clue, candidate: str
    ) -> bool:
        """
        Check if a candidate answer is compatible with existing filled cells.

        Args:
            puzzle: The crossword puzzle.
            clue: The clue we're checking.
            candidate: The candidate answer to check.

        Returns:
            True if the candidate is compatible, False otherwise.
        """
        for i, (row, col) in enumerate(clue.cells()):
            cell = puzzle.current_grid.cells[row][col]
            if cell.value is not None and cell.value != candidate[i]:
                return False
        return True
