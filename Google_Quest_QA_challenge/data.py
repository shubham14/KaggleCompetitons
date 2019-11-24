'''
Script to parse and process the data 
'''

import pandas as pd
import numpy
from glob import glob
import numpy as np
import os
import torch
from torchtext import data
from torchtext.data import Field, Dataset, Example
from os.path import join as pjoin
import re
from config import Config
import logging

cfg = Config()

LOGGER = logging.getLogger("QUEST_Question_Answering")

def read_data():
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    
    train_file_path = pjoin(cfg.base_data_dir, train_file_name)
    LOGGER.info("Loading {} from the base path {}".format(train_file_name, train_file_path))
    df_train = pd.read_csv(train_file_path)

    df_train["question_title"] = df_train.question_title.str.replace("\n", " ")
    df_train["question_body"] = df_train.question_body.str.replace("\n", " ")
    df_train["answer"] = df_train.answer.str.replace("\n", " ")
    df_train["category"] = df_train.category.str.replace("\n", " ")
    
    idx = np.arange(df_train.shape[0])
    np.random.seed(cfg.seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * cfg.VAL_RATIO)
    
    df_train.iloc[idx[val_size:], :].to_csv(pjoin(cfg.base_data_dir, "train_processed.csv"), index=False)
    df_train.iloc[idx[:val_size], :].to_csv(pjoin(cfg.base_data_dir, "val_processed.csv"), index=False)
    
    test_file_path = pjoin(cfg.base_data_dir, test_file_name)
    LOGGER.info("Loading {} from the base path {}".format(test_file_name, train_file_path))
    df_test = pd.read_csv(test_file_path)

    df_test["question_title"] = df_test.question_title.str.replace("\n", " ")
    df_test["question_body"] = df_test.question_body.str.replace("\n", " ")
    df_test["answer"] = df_test.answer.str.replace("\n", " ")
    df_test["category"] = df_test.category.str.replace("\n", " ")

    df_test.to_csv(pjoin(cfg.base_data_dir, "test_processed.csv"), index=False)
    

def tokenizer(comment):
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > cfg.MAX_CHARS):
        comment = comment[:cfg.MAX_CHARS]
    return [
        x.text for x in cfg.NLP.tokenizer(comment) if x.text != " "]


def get_dataset(lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    
    LOGGER.debug("Preparing CSV files")
    read_data()

    qa_id = data.Field(sequential=True)
    question_title = data.Field(sequential=True, fix_length=cfg.fix_length, tokenize=tokenizer, 
                                pad_first=True, lower=lower)
    question_body = data.Field(sequential=True, fix_length=cfg.fix_length, tokenize=tokenizer,
                                pad_first=True, lower=lower)
    answer = data.Field(sequential=True, fix_length=cfg.fix_length, tokenize=tokenizer, 
                        pad_first=True, lower=lower)

    LOGGER.debug("Reading train csv file")
    train, val = data.TabularDataset.splits(
        path=cfg.base_data_dir, format='csv', skip_header=True,
        train='processed_train.csv', validation='processed_val.csv',
        fields=[
            ('id', qa_id),
            ('question_title', question_title),
            ('question_body', question_body),
            ('answer', answer),
            ('question_asker_intent_understanding', data.Field(
                use_vocab=False, sequential=False)),
            ('question_body_critical', data.Field(
                use_vocab=False, sequential=False)),
            ('question_conversational', data.Field(
                use_vocab=False, sequential=False)),
            ('question_expect_short_answer', data.Field(
                use_vocab=False, sequential=False)),
            ('question_fact_seeking', data.Field(
                use_vocab=False, sequential=False)),
            ('question_has_commonly_accepted_answer', data.Field(
                use_vocab=False, sequential=False)),
            ('question_interestingness_others', data.Field(
                use_vocab=False, sequential=False)),
            ('question_interestingness_self', data.Field(
                use_vocab=False, sequential=False)),
            ('question_multi_intent', data.Field(
                use_vocab=False, sequential=False)),
            ('question_not_really_a_question', data.Field(
                use_vocab=False, sequential=False)),
            ('question_opinion_seeking', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_choice', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_compare', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_consequence', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_definition', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_entity', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_instructions', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_procedure', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_reason_explanation', data.Field(
                use_vocab=False, sequential=False)),
            ('question_type_spelling', data.Field(
                use_vocab=False, sequential=False)),
            ('question_well_written', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_helpful', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_level_of_information', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_plausible', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_relevance', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_satisfaction', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_type_instructions', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_type_procedure', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_type_reason_explanation', data.Field(
                use_vocab=False, sequential=False)),
            ('answer_well_written', data.Field(
                use_vocab=False, sequential=False)),
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path=pjoin(cfg.base_data_dir, "test_processed.csv"), format='csv', 
        skip_header=True,
        fields=[
            ('id', qa_id),
            ('question_title', question_title),
            ('question_body', question_body)
            ('answer', answer)
        ])
    LOGGER.debug("Building vocabulary...")
    question_body.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train, val, test


if __name__ == "__main__":
    train, val, test = get_dataset()
