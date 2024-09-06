import os 
import torch 

# PATHS 
PATH_LINCE_DATASET = os.environ.get("PATH_LINCE_DATASET", "./data/lince/ner")
PATH_GLUECOS_LID = os.environ.get("PATH_GLUECOS_LID_DATASET", "./data/GLUECoS/LID/Romanized")
PATH_GLUECOS_NER = os.environ.get("PATH_GLUECOS_NER_DATASET", "./data/GLUECoS/NER/Romanized")
PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")
PATH_CACHE_DATASET = os.environ.get("PATH_CACHE_DATASET", "./data/cache")

PATH_EXPERIMENTS = os.environ.get("PATH_EXPERIMENTS", "./runs")

# HARDWARE 
NUM_WORKERS = min(4, int(os.cpu_count() / 2))
AVAIL_GPUS = min(1, torch.cuda.device_count())

# HYPERPARAMS 
GLOBAL_SEED = 42

MAX_EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATE = 3e-5
WARM_RESTARTS = 50          # Restart after 50 epochs

WEIGHT_DECAY = 0
DROPOUT_RATE = 1e-3

BASE_MODEL = "bert-base-multilingual-cased"       # mBERT
MAX_SEQUENCE_LENGTH = 64
PADDING = "max_length"

LABEL2ID = {
    "NOUN": 0,   # Common nouns (e.g., dog, house)
    "VERB": 1,   # Verbs (e.g., run, eat)
    "PRON": 2,   # Pronouns (e.g., he, she)
    "ADJ": 3,    # Adjectives (e.g., big, quick)
    "ADV": 4,    # Adverbs (e.g., quickly, well)
    "ADP": 5,    # Adpositions (e.g., in, on)
    "DET": 6,    # Determiners (e.g., the, a)
    "CONJ": 7,   # Conjunctions (e.g., and, but)
    "NUM": 8,    # Numerals (e.g., one, two)
    "PART": 9,   # Particles (e.g., not, to in "to go")
    "INTJ": 10,  # Interjections (e.g., oh, wow)
    "PROPN": 11, # Proper nouns (e.g., John, Paris)
    "PUNCT": 12, # Punctuation marks (e.g., .,?!)
    "SYM": 13,   # Symbols (e.g., $, %, &)
    "X": 14,     # Other (used for words that don't fit in the above categories)
    "HI": 15,    # Hindi words that don't fall into the specific categories
    "EN": 16,    # English words that don't fall into the specific categories
    "PART_NEG": 17,
    "PRON_WH": 18,
    "OTHERS":19 
}

LID2ID = {
    "hi": 0, 
    "en": 1, 
    "rest": 2
}

GLC_NER_LABEL2ID = {
    "Other": 0,
    "B-Per": 1,
    "I-Per": 2,
    "B-Org": 3,
    "I-Org": 4,
    "B-Loc": 5,
    "I-Loc": 6
}

GLC_LID_LABEL2ID = {
    "EN": 0,
    "HI": 1,
    "OTHER": 2
}

K_CROSSFOLD_VALIDATION_SPLITS = 10

# PROJECT CONFIGURATION
PROJECT_NAME = "MetaLearning-CodeMix"
