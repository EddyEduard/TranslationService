import os

from os import environ
from dotenv import load_dotenv

# Load environment variables.

load_dotenv()


class Config:

    # Get an environment variable by name.

    def getenv(self, name):
        return environ.get(name) or os.getenv(name)
