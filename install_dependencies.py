"""Install required Python dependencies for this project.

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")


def run(command):
    """Run a command and fail fast on errors."""
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    print("Installing project dependencies...")

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])

    print("Dependency installation complete.")


if __name__ == "__main__":
    main()
