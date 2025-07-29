import pytest
from unittest.mock import Mock, patch
import os

from src.crossword.crossword import CrosswordPuzzle
from src.utils.types import Clue, Direction
from src.crossword.solvers.solver import CrosswordSolver
from src.crossword.solvers.cryptic_solver import CrypticSolver
from src.crossword.solvers.solver_factory import SolverFactory


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client for testing."""
    mock = Mock()
    # Mock the response for chat completions
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"1": ["CAT"], "2": ["COW"], "3": ["TEAR"]}'
    mock.chat.completions.create.return_value = mock_response
    return mock


@pytest.fixture
def simple_puzzle():
    """
    Creates a 5x5 puzzle with this layout:
    C A T - -
    O - E - -
    W - A - -
    - - R - -
    - - - - -
    """
    clues = [
        Clue(
            number=1,
            text="Feline friend",
            direction=Direction.ACROSS,
            length=3,
            row=0,
            col=0,
            answer="CAT"
        ),
        Clue(
            number=2,
            text="Dairy farm animal",
            direction=Direction.DOWN,
            length=3,
            row=0,
            col=0,
            answer="COW"
        ),
        Clue(
            number=3,
            text="A drop of sadness",
            direction=Direction.DOWN,
            length=4,
            row=0,
            col=2,
            answer="TEAR"
        )
    ]
    return CrosswordPuzzle(width=5, height=5, clues=clues)


@pytest.fixture
def cryptic_puzzle():
    """
    Creates a simple puzzle with cryptic clues
    """
    clues = [
        Clue(
            number=1,
            text="Feline friend arranged in cage (3)",
            direction=Direction.ACROSS,
            length=3,
            row=0,
            col=0,
            answer="CAT"
        ),
        Clue(
            number=2,
            text="Animal sounds like how (3)",
            direction=Direction.DOWN,
            length=3,
            row=0,
            col=0,
            answer="COW"
        )
    ]
    return CrosswordPuzzle(width=5, height=5, clues=clues)


class TestCrosswordSolver:
    def test_init(self, mock_client):
        solver = CrosswordSolver(client=mock_client)
        assert solver.client == mock_client
        assert solver.model == "gpt-4o"  # Default model

    def test_solve_with_answers(self, simple_puzzle, mock_client):
        solver = CrosswordSolver(client=mock_client)
        solved_puzzle = solver.solve(simple_puzzle)
        
        # Since the puzzle has answers in the clues, it should use those
        assert solved_puzzle.validate_all() is True
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[0]) == list("CAT")
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[1]) == list("COW")
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[2]) == list("TEAR")

    @patch('src.crossword.solvers.solver.CrosswordSolver._generate_batch_candidates')
    def test_solve_without_answers(self, mock_generate, simple_puzzle, mock_client):
        # Clear the answers in the clues
        for clue in simple_puzzle.clues:
            clue.answer = None
            
        # Mock the candidate generation
        mock_generate.return_value = {
            0: ["CAT"],
            1: ["COW"],
            2: ["TEAR"]
        }
        
        solver = CrosswordSolver(client=mock_client)
        solved_puzzle = solver.solve(simple_puzzle)
        
        # Check that the solver called generate_batch_candidates
        mock_generate.assert_called_once()
        
        # Check the puzzle was solved correctly
        assert solved_puzzle.validate_all() is True
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[0]) == list("CAT")
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[1]) == list("COW")
        assert solved_puzzle.get_current_clue_chars(simple_puzzle.clues[2]) == list("TEAR")

    def test_determine_difficulty(self, simple_puzzle, mock_client):
        solver = CrosswordSolver(client=mock_client)
        difficulty = solver._determine_difficulty(simple_puzzle)
        assert difficulty == "easy"
        
        # Test with more clues
        # Create a puzzle with more clues
        large_clues = []
        for i in range(25):
            large_clues.append(
                Clue(
                    number=i+1,
                    text=f"Clue {i+1}",
                    direction=Direction.ACROSS if i % 2 == 0 else Direction.DOWN,
                    length=3,
                    row=i % 10,
                    col=i % 10,
                    answer="ABC"
                )
            )
        large_puzzle = CrosswordPuzzle(width=15, height=15, clues=large_clues)
        difficulty = solver._determine_difficulty(large_puzzle)
        assert difficulty == "hard"


class TestCrypticSolver:
    def test_init(self, mock_client):
        solver = CrypticSolver(client=mock_client)
        assert solver.client == mock_client
        assert solver.model == "gpt-4o"  # Default model
        
    def test_analyze_cryptic_clue(self, mock_client):
        solver = CrypticSolver(client=mock_client)
        
        # Test anagram indicator detection
        analysis = solver._analyze_cryptic_clue("Confused doctor in charge (7)")
        assert "anagram" in analysis["wordplay_types"]
        assert "confused" in analysis["indicators"]
        
        # Test hidden word indicator detection
        analysis = solver._analyze_cryptic_clue("Leader hidden in presidential election (4)")
        assert "hidden" in analysis["wordplay_types"]
        assert "hidden" in analysis["indicators"]
        
        # Test reversal indicator detection
        analysis = solver._analyze_cryptic_clue("Return oriental drink (3)")
        assert "reversal" in analysis["wordplay_types"]
        assert "return" in analysis["indicators"]


class TestSolverFactory:
    def test_create_solver_standard(self, simple_puzzle, mock_client):
        solver = SolverFactory.create_solver(
            puzzle=simple_puzzle,
            client=mock_client,
            verbose=True
        )
        assert isinstance(solver, CrosswordSolver)
        assert not isinstance(solver, CrypticSolver)
        
    def test_create_solver_cryptic(self, cryptic_puzzle, mock_client):
        solver = SolverFactory.create_solver(
            puzzle=cryptic_puzzle,
            client=mock_client,
            verbose=True
        )
        assert isinstance(solver, CrypticSolver)
        
    def test_is_cryptic_puzzle(self):
        # Test with cryptic indicators
        cryptic_clues = [
            Clue(
                number=1,
                text="Confused about Eastern drink (3)",
                direction=Direction.ACROSS,
                length=3,
                row=0, col=0, answer=None
            ),
            Clue(
                number=2,
                text="Hidden in plain sight (5)",
                direction=Direction.DOWN,
                length=5,
                row=0, col=0, answer=None
            )
        ]
        cryptic_puzzle = CrosswordPuzzle(width=5, height=5, clues=cryptic_clues)
        assert SolverFactory._is_cryptic_puzzle(cryptic_puzzle) is True
        
        # Test without cryptic indicators
        standard_clues = [
            Clue(
                number=1,
                text="Feline pet (3)",
                direction=Direction.ACROSS,
                length=3,
                row=0, col=0, answer=None
            ),
            Clue(
                number=2,
                text="Dairy animal (3)",
                direction=Direction.DOWN,
                length=3,
                row=0, col=0, answer=None
            )
        ]
        standard_puzzle = CrosswordPuzzle(width=5, height=5, clues=standard_clues)
        assert SolverFactory._is_cryptic_puzzle(standard_puzzle) is False
