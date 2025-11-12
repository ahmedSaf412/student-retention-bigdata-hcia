# config/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the path to the data file
DATA_PATH = os.getenv("DATA_PATH", "data/student_sample.csv")