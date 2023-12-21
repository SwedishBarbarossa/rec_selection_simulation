
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install -r required.txt

# Run tests
pytest

# Run main.py
python3 main.py
