from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path=r'D:\PROGRAMING\CodeBluepdf\basic-no-web-app\.env')

# Retrieve API key
AI71_API_KEY = os.getenv('AI71_API_KEY')

if AI71_API_KEY is None:
    raise ValueError("API key not found. Please check your .env file.")
else:
    print("API key successfully loaded.")
