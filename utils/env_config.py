import os
from dotenv import load_dotenv

def get_hf_user():
    """
    get hf username from env, with fallback to default
    """
    load_dotenv()
    return os.getenv("HF_USERNAME", "AngelRaychev")

def get_hf_token():
    """
    get hf token from env if available
    """
    load_dotenv()
    return os.getenv("HF_TOKEN")