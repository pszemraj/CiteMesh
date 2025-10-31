"""
Entry point for python -m citemesh execution.

This is just a thin wrapper - all CLI logic lives in citemesh.cli
"""

from citemesh.cli import main

if __name__ == "__main__":
    main()
