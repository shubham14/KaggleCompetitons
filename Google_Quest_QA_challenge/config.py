'''
configuration files for the constants
'''
import spacy
from os.path import join as pjoin

class Config:
    NLP = spacy.load('en_core_web_sm')
    MAX_CHARS = 20000
    dirname = r"C:\\Users\\Shubham\\Desktop\\KaggleCompetitons"
    base_data_dir = pjoin(dirname, 'data\\Google_QUEST_Challenge')
    seed = 42
    VAL_RATIO = 0.2 # validation set ratio
    fix_length = 200 # number of charecters in the field 