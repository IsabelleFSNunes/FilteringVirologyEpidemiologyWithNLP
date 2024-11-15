""" Definition of libraries and packages""" 
import preprocessing_data
import processing_data
import utils 

import os

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import torch
import json

from transformers import AdamW
from transformers import BertTokenizer, BertModel

import pandas as pd
import numpy as np

from tqdm import tqdm
import time

# to start the execution time 
start_time = time.time()

# Check for GPU availability
CURRENT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {CURRENT_DEVICE}')

# Create data loaders
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

# Checkpoint path
CHECKPOINT_PATH = "model_checkpoint.pt"

def main():
    '''
        The main function that orchestrates the complete filtering task.
    '''
    ## Model 
    model= preprocessing_data.load_llm_model(CURRENT_DEVICE, n_labels= 2)
    
    model.classifier.apply(processing_data.initialize_weights)
    model.train()

    # to protect the layers to not trained all parameters
    for param in model.bert.parameters():
        param.requires_grad = False


    ## Database
    train_df = preprocessing_data.load_virology_paper("./DB/input/collection_with_abstracts.csv")
    train_loader = preprocessing_data.create_data_loaders(train_df, MAX_LENGTH, TRAIN_BATCH_SIZE)

    descriptors_json = preprocessing_data.load_descriptors_json("./DB/input/descriptors.json")

    ## Optimization
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Training for 3 epochs
    for epoch in range(3):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(CURRENT_DEVICE)
            attention_mask = batch['attention_mask'].to(CURRENT_DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            optimizer.step()

            preds = torch.argmax(outputs.logits, dim=1)
    model.eval()

    os.makedirs('./modelteste', exist_ok=True)
    utils.save_onnx("./model/bert_sequence_classification.onnx", model, input_ids, attention_mask)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    descritor_embeddings = processing_data.get_embeddings_batch(descriptors_json['deep learning'], tokenizer, model, batch_size=16)


    ### Task 01: Filter the relevant papers, Deep Learning in Virology/Epidemiology 
    # -------------------------------------------------------
    train_df['is_relevant'] = train_df['Concat Text'].apply(lambda x: processing_data.is_relevant(x, descritor_embeddings, tokenizer=tokenizer, model=model, threshold=0.85))

    relevant_papers = train_df[train_df['is_relevant']]

    embeddings_relevant = processing_data.get_embeddings_batch( list(relevant_papers["Concat Text"]) , tokenizer=tokenizer, model=model, batch_size=1)
    

    ### Task 02: Classifying the clusters
    # -------------------------------------------------------   
    relevant_papers['methods_used'] = processing_data.classify_semantic_methods(embeddings_relevant, relevant_papers, "./DB/output/figure_kmeans.png")
    print("Methods used was classified.")


    ### Task 03: Verify the method used
    # -------------------------------------------------------
    relevant_papers['methods_name'] = processing_data.extract_method_names(relevant_papers['Concat Text'])

    

    path_output = "./DB/output/relevant_papers.csv"
    relevant_papers.to_csv(path_output)
    print(f"The {path_output} saved sucessfully.")
    
    relevant_papers.loc[relevant_papers['methods_used'] == 'computer vision and text mining', 'methods_used'] = 'both'

    print('---------------------------------------------------------------------------------------------')
    print(f'Total of relevant papers (task 1): {len(relevant_papers)}')
    print('---------------------------------------------------------------------------------------------')
    # Lista dos métodos e nomes dos arquivos CSV para salvar cada subset
    method_labels = {
        "computer vision": "./DB/output/computer_vision.csv",
        "text mining": "./DB/output/text_mining.csv",
        "both": "./DB/output/both.csv",
        "other": "./DB/output/other.csv"
    }
    size_filtered = []
    for method, filename in method_labels.items():
        filtered_papers = relevant_papers[relevant_papers['methods_used'] == method]
        size_filtered.append(len(filtered_papers))

        filtered_papers.drop(['Concat Text', 'is_relevant', 'cluster_label'], axis=1, inplace=True)
        filtered_papers.to_csv(filename, index=False)
        out_print = f'Number of paper of {method}: {len(filtered_papers)} \t Path of file {filename}'
        print(out_print.center(90))
        
    
        
    print('---------------------------------------------------------------------------------------------')
    print(f'Summary statistics: ')
    print('---------------------------------------------------------------------------------------------')
    print(f'Total input: {len(train_df)} (100 %).')
    relevant_percent = (100 * len(relevant_papers)/len(train_df))
    print(f'Relevant to DL in Virology/Epidemiology: {len(relevant_papers)} ({relevant_percent:.2f} % of total). ')
    print('---------------------------------------------------------------------------------------------')
    for size, (method, filename) in zip(size_filtered,  method_labels.items()): 
        out_print = f'Papers of {method}: {size} ({(relevant_percent * size/len(relevant_papers)):.2f} % of total) \t ({(100 * size/len(relevant_papers)):.2f} % of relevant papers) '
        print(out_print.rjust(90))


if __name__ == "__main__":
    main()


end_time = time.time()
execution_time = (end_time - start_time) / 60.0
print(f"Duration of execution: {execution_time:.2f} min")