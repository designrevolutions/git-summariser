from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

nebius_api_key = os.getenv("NEBIUS_API_KEY")

print(nebius_api_key)