"""
Run all crossword puzzles in sequence to test the solver.
"""

import os
import argparse
import time
from dotenv import load_dotenv
from tabulate import tabulate

from main import solve_puzzle

# Load environment variables from .env file
load_dotenv()


def run_all_puzzles(model="gpt-4o", verbose=False):
    """
    Run all puzzles in sequence and report results.

    Args:
        model: The OpenAI model to use.
        verbose: Whether to print verbose output.
    """
    puzzle_files = [
        "data/easy.json",
        "data/medium.json",
        "data/hard.json",
        "data/cryptic.json",
    ]

    results = []

    for puzzle_file in puzzle_files:
        print(f"\n\n{'='*50}")
        print(f"Solving: {puzzle_file}")
        print(f"{'='*50}")

        start_time = time.time()
        solved_puzzle, _ = solve_puzzle(
            puzzle_file,
            verbose=verbose,
            model=model,
        )
        end_time = time.time()
        solve_time = end_time - start_time

        # Check if solution is complete
        is_complete = solved_puzzle.validate_all()

        # Print the solved puzzle
        print("\n--- Solved Puzzle ---")
        print(solved_puzzle)

        # Store results for summary
        puzzle_name = os.path.basename(puzzle_file).replace(".json", "")
        grid_size = f"{solved_puzzle.width}x{solved_puzzle.height}"
        completion_status = "✅" if is_complete else "❌"
        results.append(
            [
                puzzle_name,
                len(solved_puzzle.clues),
                grid_size,
                f"{solve_time:.2f}s",
                completion_status,
            ]
        )

    # Print summary table
    print("\n\n")
    print("Summary of Results:")
    print(
        tabulate(
            results,
            headers=["Puzzle", "Clues", "Grid Size", "Time", "Complete?"],
            tablefmt="grid",
        )
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run all crossword puzzles in sequence."
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

    run_all_puzzles(model=args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
