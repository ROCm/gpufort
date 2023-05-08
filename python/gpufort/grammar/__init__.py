
import os

GRAMMAR_DIR = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_PATH = os.path.join(GRAMMAR_DIR, "grammar.py")

from .grammar import *
