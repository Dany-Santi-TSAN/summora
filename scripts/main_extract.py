"""
Summora - Module d'Extraction de Contenu
Extraction de topics, actions, décisions et insights depuis transcriptions
Usage: python main_extract.py transcription.txt --topics --actions --decisions
"""
import argparse
import json
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Imports spécialisés avec fallbacks

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords, opinion_lexicon
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
