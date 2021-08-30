import os
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# NLP
import gluonnlp as nlp


# Transformer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel, TFBertModel, TFRobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSequenceClassification


class TrainDataset_large(Dataset):

    def __init__(self, df, tokenizer_roberta_large, MAX_LEN):
        self.df_data = df
        self.tokenizer_roberta_large = tokenizer_roberta_large
        self.MAX_LEN = MAX_LEN
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.iloc[index,0]
        encoded_dict = self.tokenizer_roberta_large(
          text = sentence,
          add_special_tokens = True, 
          max_length = self.MAX_LEN,
          padding='max_length',
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        target = torch.tensor(self.df_data.iloc[index,1])
        sample = (padded_token_list, token_type_id , att_mask, target)
        return sample
    def __len__(self):
        return len(self.df_data)

class TestDataset_large(Dataset):
    
    def __init__(self, df, tokenizer_roberta_large, MAX_LEN):
        self.df_data = df
        self.tokenizer_roberta_large = tokenizer_roberta_large
        self.MAX_LEN = MAX_LEN
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.loc[index, '발화']
        encoded_dict = self.tokenizer_roberta_large(
          text = sentence,
          add_special_tokens = True, 
          max_length = self.MAX_LEN,
          padding='max_length',
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, token_type_id , att_mask)
        return sample
    def __len__(self):
        return len(self.df_data)


class TrainDataset_small(Dataset):
    
    def __init__(self, df, tokenizer_roberta_small,MAX_LEN):
        self.df_data = df
        self.tokenizer_roberta_small = tokenizer_roberta_small
        self.MAX_LEN = MAX_LEN
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.iloc[index,0]
        encoded_dict = self.tokenizer_roberta_small(
          text = sentence,
          add_special_tokens = True, 
          max_length = self.MAX_LEN,
          padding='max_length',
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        target = torch.tensor(self.df_data.iloc[index,1])
        sample = (padded_token_list, token_type_id , att_mask, target)
        return sample
    def __len__(self):
        return len(self.df_data)

class TestDataset_small(Dataset):
    
    def __init__(self, df, tokenizer_roberta_small,MAX_LEN):
        self.df_data = df
        self.tokenizer_roberta_small = tokenizer_roberta_small
        self.MAX_LEN = MAX_LEN
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.loc[index, '발화']
        encoded_dict = self.tokenizer_roberta_small(
          text = sentence,
          add_special_tokens = True, 
          max_length = self.MAX_LEN,
          padding='max_length',
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, token_type_id , att_mask)
        return sample
    def __len__(self):
        return len(self.df_data)
