# LLM Crossword Solver

An intelligent crossword puzzle solver powered by GPT-4o, capable of solving both standard and cryptic crosswords with specialized strategies. This implementation always uses the LLM to generate answers rather than relying on pre-provided solutions.

## Features

- **Pure LLM-based Solving**: Uses GPT-4o to generate answers for all clues, regardless of whether answers are provided in the data
- **Intelligent Puzzle Type Detection**: Automatically detects puzzle types (standard vs. cryptic) based on clue patterns
- **Specialized Solvers**: Uses targeted strategies for different puzzle types
- **Factory Pattern**: Dynamically selects the appropriate solver implementation
- **Constraint Satisfaction**: Resolves conflicts between intersecting clues
- **Batch Processing**: Efficiently handles clues in batches to optimize API calls
- **Performance Benchmarking**: Tracks solving times and completion rates across different puzzle difficulties

## How It Works - Step by Step

The crossword solver follows a systematic approach to solving puzzles:

1. **Puzzle Loading**: Load the puzzle from a JSON file
2. **Solver Selection**:
   - The `SolverFactory` analyzes the puzzle characteristics
   - Detects if it's a standard or cryptic puzzle based on clue patterns
   - Creates the appropriate solver (standard `CrosswordSolver` or specialized `CrypticSolver`)
3. **Puzzle Reset**: Clear any existing answers to start fresh
4. **LLM-based Solving**:
   - The solver always uses the LLM to generate answers, even if answers are provided in the data
   - Difficulty determination based on grid size and clue characteristics
   - Batch processing of clues (5 clues at a time) to optimize API calls
5. **Answer Generation**:
   - System prompts tailored to the puzzle difficulty and type
   - For cryptic puzzles, specialized prompts that understand wordplay techniques
   - Response parsing to extract candidate answers of the correct length
6. **Constraint Satisfaction**:
   - Sort clues by the number of candidates (most constrained first)
   - Backtracking algorithm to find a consistent solution
   - Check compatibility between intersecting answers
7. **Solution Validation**: Ensure all cells are filled and constraints are satisfied
8. **Result Presentation**: Display the completed grid and solving statistics

### Key Components

1. **CrosswordPuzzle Class**: Represents the crossword puzzle structure and state
   - Handles grid management, clue placement, and validation
   - Maintains history for undo operations
   
2. **CrosswordSolver Class**: Base solver for standard crosswords
   - Uses LLM to generate candidate answers for clues
   - Applies difficulty-based prompting strategies
   - Resolves conflicts between intersecting answers

3. **CrypticSolver Class**: Specialized solver for cryptic crosswords
   - Extends the base solver with cryptic-specific prompting
   - Analyzes wordplay components (anagrams, hidden words, etc.)
   - Uses specialized prompts that understand cryptic techniques

4. **SolverFactory Class**: Creates appropriate solver based on puzzle characteristics
   - Analyzes clue patterns to identify cryptic puzzles
   - Returns either standard or cryptic solver


### Solver Types and Strategies

#### Standard Crosswords

For standard crosswords, the solver:
- Uses direct prompting focused on general knowledge
- Provides difficulty context to the LLM
- Prioritizes common words and phrases
- System prompt example (easy difficulty):
  ```
  You are helping to solve a crossword puzzle.
  For each clue, provide the most likely answer that fits the length requirement.
  ```

#### Cryptic Crosswords

For cryptic crosswords, the solver employs specialized techniques:
- Analyzes clue components (definition part vs. wordplay part)
- Identifies cryptic indicators (anagrams, hidden words, etc.)
- Uses expert prompts with examples of cryptic solving techniques
- Evaluates candidate answers based on wordplay understanding
- System prompt example:
  ```
  You are an expert cryptic crossword solver with decades of experience. 
  Cryptic clues contain wordplay and often have misleading surface readings.
  [... detailed explanation of cryptic techniques ...]
  ```

### Performance Benchmarks

Based on testing with provided puzzles:

| Puzzle Type | Grid Size | Clues | Completion Rate | Avg. Solving Time |
|-------------|-----------|-------|----------------|-------------------|
| Easy        | 5x5       | 3     | 100%           | ~1.1 seconds      |
| Medium      | 7x7       | 5     | 100%           | ~0.9 seconds      |
| Hard        | 13x13     | 23    | Partial        | ~4.2 seconds      |
| Cryptic     | 15x15     | 30    | Partial        | ~4.7 seconds      |

The solver successfully completes easy and medium puzzles but struggles with the larger, more complex puzzles due to the combinatorial explosion of constraints.

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/lcardno10/llm-crossword.git
cd llm-crossword
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
cp .env.example .env  # Copy the example env
# Edit the .env and replace with your Azure OpenAI API details:
# OPENAI_API_VERSION=your-api-version
# AZURE_OPENAI_ENDPOINT=your-endpoint
# AZURE_OPENAI_API_KEY=your-api-key
```

## Usage

### Solving a Single Puzzle

To solve a specific puzzle:

```bash
python main.py --puzzle data/easy.json --verbose
```

Available options:
- `--puzzle`: Path to the puzzle JSON file (default: data/easy.json)
- `--model`: OpenAI model to use (default: gpt-4o)
- `--verbose`: Print detailed output during solving

### Solving Multiple Puzzles

To solve all puzzles in sequence and see a comparison:

```bash
python run_all_puzzles.py --verbose
```

This will run through all puzzles (easy, medium, hard, and cryptic) and display a summary table with results.

## Puzzle Files

The system supports JSON crossword files with the following structure:

```json
{
  "width": 5,
  "height": 5,
  "clues": [
    {
      "number": 1,
      "text": "Feline friend",
      "direction": "across",
      "length": 3,
      "row": 0,
      "col": 0,
      "answer": "CAT"
    }
  ]
}
```

Puzzle difficulty levels:
- **Easy**: Simple clues, small grid (5x5)
- **Medium**: Moderate difficulty clues, medium grid (7x7)
- **Hard**: Challenging clues, larger grid (13x13)
- **Cryptic**: Wordplay-based clues, complex grid (15x15)

## Key Changes Made

The primary modifications to make the solver truly use the LLM:

1. **Forced LLM Usage**: Modified `solver.py` to always use the LLM for answer generation
   ```python
   # Always use LLM to solve the puzzle even if answers are provided in the data
   if verbose:
       logging.info("Using LLM to generate answers for all clues.")
   return self._solve_with_llm(puzzle, verbose)
   ```

2. **Removed Pre-filled Answer Usage**: Eliminated code that would use pre-provided answers in the backtracking algorithm
   ```python
   # The following code was removed:
   # if clue.answer and not clue.answered:
   #    if verbose:
   #        logging.info(f"Using provided answer for clue {clue_idx+1}: {clue.answer}")
   #    puzzle.set_clue_chars(clue, list(clue.answer))
   #    return self._backtrack(puzzle, sorted_clue_indices, candidates, index + 1, verbose)
   ```

3. **Enhanced Cryptic Solver**: Improved prompting for cryptic crosswords with detailed technique explanations

4. **Batch Processing**: Optimized API usage by processing clues in small batches (5 clues at a time)

5. **Retry Logic**: Added robust error handling and retry mechanisms for API calls

## Project Structure

```
llm_crossword/
├── src/                          # Source code
│   ├── crossword/                # Main crossword package
│   │   ├── __init__.py           # Package initialization
│   │   ├── crossword.py          # Puzzle representation and state management
│   │   ├── create.py             # Puzzle creation utilities
│   │   ├── download.py           # Puzzle downloading utilities
│   │   ├── solvers/              # Solver implementations package
│   │   │   ├── __init__.py       # Solver package initialization
│   │   │   ├── solver.py         # Base solver (modified to always use LLM)
│   │   │   ├── cryptic_solver.py # Cryptic-specific solver
│   │   │   └── solver_factory.py # Dynamic solver selection logic
│   │   └── exceptions/           # Custom exceptions package
│   │       ├── __init__.py       # Exceptions package initialization
│   │       └── exceptions.py     # Custom exception definitions
│   └── utils/                    # Utility functions package
│       ├── __init__.py           # Utils package initialization
│       ├── types.py              # Data type definitions (Clue, Direction, etc.)
│       └── utils.py              # Puzzle loading and manipulation utilities
├── tests/                        # Test suite
│   ├── __init__.py               # Test package initialization
│   ├── test_crossword.py         # Tests for crossword logic and puzzle state
│   └── test_solvers.py           # Tests for all solver implementations
├── data/                         # Puzzle data files
│   ├── easy.json                 # Easy puzzle (5x5, 3 clues)
│   ├── medium.json               # Medium puzzle (7x7, 5 clues)
│   ├── hard.json                 # Hard puzzle (13x13, 23 clues)
│   └── cryptic.json              # Cryptic puzzle (15x15, 30 clues)
├── main.py                       # Main entry point for single puzzle solving
├── run_all_puzzles.py            # Script to benchmark all puzzles
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

## Future Improvements

Potential enhancements for the solver:

1. **Advanced Constraint Propagation**: Implement more sophisticated constraint satisfaction techniques
2. **Multi-step Solving**: Add iterative solving approaches with feedback loops to the LLM
3. **Answer Confidence Scoring**: Rank candidate answers by confidence level
4. **Dynamic Temperature Adjustment**: Vary LLM temperature based on puzzle difficulty
5. **Human-in-the-Loop**: Add capability for human intervention on difficult clues
6. **Custom Embedding**: Create specialized embeddings for crossword vocabulary

## Testing

Run the test suite:
```bash
pytest
```

## License

This challenge is proprietary and confidential. Do not share or distribute.
