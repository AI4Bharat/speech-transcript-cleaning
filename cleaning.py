import sys
import string
import re
from pathlib import Path
import os
import regex

import numpy as np
import pandas as pd
from tqdm import tqdm
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.sentence_tokenize import sentence_split

from num_to_word import num_to_word

from joblib import Parallel, delayed

REMOVE_NUKTAS=False

def normalize_sentence(sentence, lang_code):
    '''
    Perform NFC -> NFD normalization for a sentence and a given language
    sentence: string
    lang_code: language code in ISO format
    '''
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence

def normalize_sentences(sentences, lang_code):
    '''
    Perform NFC -> NFD normalization for a list of sentences and a given language
    sentence: list of strings
    lang_code: language code in ISO format
    '''
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    normalized_sentences = [normalizer.normalize(sentence) for sentence in tqdm(sentences)]
    return normalized_sentences

def convert_num_to_word_sentence(line, lang_code):
    '''
    Convert all indic numbers to words for a given sentence
    line: string
    lang_code: language code in ISO format
    '''
    new_line = ''
    line= re.sub('[!@#$%।,₹]', '', line)
    line= re.sub('[:.]', ' ', line)
    for word in line.rstrip().lstrip().split(' '):
        if word =='':
            continue
        try:
            new_line += (num_to_word(word, lang=lang_code, separator=' ')) + ' '
        except:
            new_line += word + ' '
    new_line = new_line.rstrip().lstrip()
    return new_line

def indic_language_identifier(text, lang_code):
    if lang_code == 'as':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Bengali}+(?:\P{L}+\p{Bengali}+)*\P{L}*', text))
    elif lang_code == 'bn':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Bengali}+(?:\P{L}+\p{Bengali}+)*\P{L}*', text))
    elif lang_code == 'gu':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Gujarati}+(?:\P{L}+\p{Gujarati}+)*\P{L}*', text))
    elif lang_code == 'hi':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Devanagari}+(?:\P{L}+\p{Devanagari}+)*\P{L}*', text))
    elif lang_code == 'kn':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Kannada}+(?:\P{L}+\p{Kannada}+)*\P{L}*', text))
    elif lang_code == 'ml':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Malayalam}+(?:\P{L}+\p{Malayalam}+)*\P{L}*', text))
    elif lang_code == 'mr':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Devanagari}+(?:\P{L}+\p{Devanagari}+)*\P{L}*', text))
    elif lang_code == 'or':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Oriya}+(?:\P{L}+\p{Oriya}+)*\P{L}*', text))
    elif lang_code == 'pa':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Gurmukhi}+(?:\P{L}+\p{Gurmukhi}+)*\P{L}*', text))
    elif lang_code == 'sa':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Devanagari}+(?:\P{L}+\p{Devanagari}+)*\P{L}*', text))
    elif lang_code == 'ta':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Tamil}+(?:\P{L}+\p{Tamil}+)*\P{L}*', text))
    elif lang_code == 'te':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Telugu}+(?:\P{L}+\p{Telugu}+)*\P{L}*', text))
    elif lang_code == 'ur':
        check_indic = bool(regex.fullmatch(r'\P{L}*\p{Arabic}+(?:\P{L}+\p{Arabic}+)*\P{L}*', text))
    return check_indic


def cleaning_pipeline(input_file, output_file, lang_code):
    '''
    Standardize speech transcripts for a dataset
    input_file: file containing sentences (one sentence per line)
    output_file: path to output file
    lang_code: language code in ISO format
    '''

    # Load character set from dictionary file
    dict_file = f"dicts/{lang_code}.dict.txt"
    dict_df = pd.read_csv(dict_file, header=None, sep=' ')
    dict_characters = ''.join(dict_df[0].to_list())
    dict_characters += ' '
    print("Dictionary Loaded.")

    print("Standardizing..")
    count_no_ood = 0 # Count number of out-of-dictionary sentences
    total_sents = 0 # count total number of sentences
    with open(input_file, "r") as read_fp, \
        open(output_file, "w") as write_fp:
        # batchwrite to disk for faster IO
        batchsize = 1000
        batch = []
        for idx, sentence in enumerate(tqdm(read_fp)):
            ## Remove punctuations (add danda to punctuation list)
            sentence = sentence.translate(str.maketrans('', '', string.punctuation+'।'))
            ## Convert num to word (Don't do for speech transcripts! used for LM text cleaning)
            # sentence = convert_num_to_word_sentence(sentence, lang_code)
            ## Normalize sentence
            sentence = normalize_sentence(sentence, lang_code)
            # Drop sentence having OOD characters (Don't drop for ASR, using it to calculate stats)
            # Used for LM text cleaning
            matched_list = [char in dict_characters for char in sentence]
            no_ood = all(matched_list)
            if no_ood:
                count_no_ood += 1 # no OOD characters in sentence
            # Adding sentence to batch
            batch.append(sentence+"\n")
            total_sents += 1
            # write batch to disk
            if len(batch) == batchsize:
                write_fp.writelines(batch)
                batch = []
        # write last bacth to disk
        write_fp.writelines(batch)
    
    # Print stats
    print(f"In Dictionary sentences: {count_no_ood}/{total_sents}")
    print("Complete!")

def crisp_cleaning_pipeline(inline, lang_code):
    inline = inline.translate(str.maketrans('', '', string.punctuation+'।'))
    inline = convert_num_to_word_sentence(inline, lang_code)
    inline = normalize_sentence(inline, lang_code)
    inlang = indic_language_identifier(inline, lang_code)
    if inlang:
        return inline
    return None

if __name__ == "__main__":

    input_file = sys.argv[1] 
    output_file = sys.argv[2]
    language = sys.argv[3]

    dict_file = f"dicts/{language}.dict.txt"
    dict_df = pd.read_csv(dict_file, header=None, sep=' ')
    dict_characters = ''.join(dict_df[0].to_list())
    dict_characters += ' '
    print("Dictionary Loaded.")

    # cleaning_pipeline(input_file, output_file, language)
    with open(input_file, "r") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    lines = [line for line in lines if line != '']

    split_lines = []
    for line in lines:
        split_lines.extend(sentence_split(line, language))

    lines = Parallel(n_jobs=8)(delayed(crisp_cleaning_pipeline)(line, language) for line in tqdm(split_lines))
    with open(output_file, "w") as f:
        for line in lines:
            if line is not None:
                f.write(line+'\n')