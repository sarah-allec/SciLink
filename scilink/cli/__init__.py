"""SciLink CLI interface."""

from .workflows import main as experimental_novelty_main
from .agents import add_agent_args

__all__ = ['experimental_novelty_main', 'add_agent_args']
