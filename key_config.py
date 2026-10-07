import os
from os.path import dirname, join

from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

LLM_SERVER = os.getenv("LLM_SERVER")
LLM_TOKEN = os.getenv("LLM_TOKEN")
