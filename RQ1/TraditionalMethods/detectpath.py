import os

# Get the absolute path of the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Define the project base path (BASE_PATH)
# Set this to your project root directory
BASE_PATH = os.path.dirname(os.path.dirname(current_path))

# Set JOERN_PATH to the directory of your joern-cli installation
JOERN_PATH = "/path/to/joern/joern-cli/"