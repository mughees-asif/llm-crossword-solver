"""
Main script to solve crossword puzzles using LLM.
"""

import os
import argparse
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.utils.utils import load_puzzle
from src.crossword.solvers.solver_factory import SolverFactory

# Load environment variables from .env file
load_dotenv()


def initialize_openai_client():
    """Initialize the OpenAI client."""
    return AzureOpenAI(
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def solve_puzzle(puzzle_file, verbose=False, model="gpt-4o"):
    """
    Solve a crossword puzzle.

    Args:
        puzzle_file: Path to the puzzle file.
        verbose: Whether to print verbose output.
        model: The OpenAI model to use.

    Returns:
        The solved puzzle and the time taken.
    """
    # Load the puzzle
    start_time = time.time()
    puzzle = load_puzzle(puzzle_file)

    if verbose:
        print(f"Loaded puzzle with {len(puzzle.clues)} clues")
        print(f"Grid size: {puzzle.width}x{puzzle.height}")

    # Initialize the client
    client = initialize_openai_client()

    # Use the factory to create an appropriate solver
    solver = SolverFactory.create_solver(
        puzzle=puzzle, client=client, model=model, verbose=verbose
    )

    # Solve the puzzle
    solved_puzzle = solver.solve(puzzle, verbose=verbose)

    end_time = time.time()
    solve_time = end_time - start_time

    return solved_puzzle, solve_time


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Solve a crossword puzzle using LLM."
    )
    parser.add_argument(
        "--puzzle", 
        type=str, 
        default="data/easy.json", 
        help="Path to the puzzle file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    print(f"Solving puzzle: {args.puzzle}")
    solved_puzzle, solve_time = solve_puzzle(
        puzzle_file=args.puzzle,
        verbose=args.verbose,
        model=args.model,
    )

    print("\n--- Solved Puzzle ---")
    print(solved_puzzle)
    print(f"\nTime taken: {solve_time:.2f} seconds")
    print(
        f"Is complete: {solved_puzzle.validate_all()}"
    )


if __name__ == "__main__":
    main()
