import json
import nltk
from database_class import VirologyPapersDataset

import pandas as pd
import numpy as np
import re 

from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split


def load_virology_paper(filename):
    '''
        Load the specific dataset of article attributes.         
        inputs: 
            filename (str): The file name of input database, considering the path. 

        output: 
            train_df (pandas.DataFrame): The preprocessed training dataset.
            test_df (pandas.DataFrame): The preprocessed test dataset.
    '''
    # print("Inside load sentiment data")
    complete_df= pd.read_csv(filename, encoding='unicode_escape')

    complete_df= complete_df.fillna('')
    complete_df_one_string_column = pd.DataFrame(complete_df[['Title', 'Authors', 'Citation','Abstract']].agg(''.join, axis=1), columns=['Concat Text'] )

    ## split database
    # 20%: test and 80%: train
    train_df, test_df = train_test_split(complete_df_one_string_column, test_size= 0.2, random_state= 42)

    train_df.drop_duplicates()
    test_df.drop_duplicates()

    return train_df, test_df

def create_data_loaders(train_df, test_df, max_length, batch_size):
    ''' 
        Create PyTorch data loaders for the classification of papers task, considering the BertTokenizer (bert-base-uncased).

    input:
        train_df (pandas.DataFrame): The preprocessed training dataset.
        test_df (pandas.DataFrame): The preprocessed test dataset.
        max_length (int): The maximum length of the input sequences.
        batch_size (int): The batch size for the data loaders.

    output:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
    '''
    # Load model directly
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = VirologyPapersDataset(train_df, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = VirologyPapersDataset(test_df, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def load_llm_model( current_device, n_labels=2 ):
    '''
        Load the BERT model, For Sequence Classification, considering bert-base-uncased.
    
    input:
        current_device (str): Flag to identify the specific execution, cpu or cuda.
        n_labels (int): Number of labels that could be classified in the model.
    output: 
        model (transformers.BertForSequenceClassification): The loaded classification model.
    '''
    # Load model directly
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_labels)
    model.to(current_device)

    return model

###  TODO: Verificar se mantenho no codigo geral 
def preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    '''
        Preprocess a string.
    
        input:
            text (str): Name of column containing text
            lst_stopwords (list): List of stopwords to remove
            flg_stemm (bool): Whether stemming is to be applied
            flg_lemm (bool):  Whether lemmitisation is to be applied

        output: 
            cleaned text
    '''

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

def load_descriptors_json(filename):
    ''' 
        Load json file with descriptors that will be applied in this specific context

        inputs: 
            filename (str): The file name of input (json) database, considering the path. 

    '''
    # json with descriptors
    with open(filename, 'r') as file:
        descriptors_json = json.load(file)
    
    return descriptors_json
    