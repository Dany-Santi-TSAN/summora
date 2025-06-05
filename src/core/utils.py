"""
Utilitaires généraux pour SMA : Summora Meeting Analyzer
Configuration NLTK, validation des fichiers audio et constantes
"""
import nltk
from pathlib import Path
from typing import Set, Optional
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
