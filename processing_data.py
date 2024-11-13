import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score

import spacy

import preprocessing_data

# Check for GPU availability
CURRENT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_embeddings_batch(texts, tokenizer, model, batch_size=16):
    '''
        Get embeddings with to model pre-defined in the main. This type of embedding considering the Logits applied in the BertForSequenceClassification. 

        input:
            texts (list str): Input text, data or descriptor to apply the model. 
            tokenizer (transformers.BertTokenizer): The loaded Tokenizer to Bert LLM.
            model (transformers.BertForSequenceClassification): The loaded classification model. 
            batch_size (int): Size of batch applied in embedding.

        output: 
            embeddings (torch.Tensor): A Tensor with the list of embedding applied to model choosen. 
    '''
    embeddings = []

    for i in range(0, len(texts), batch_size):
        # Extrair o lote de textos atual
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize considering the GPU 
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(CURRENT_DEVICE)
        
        # No consider the gradients to optimize process 
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        embeddings.append(logits.cpu())
  

    # Join the embeddings of all batchs applied
    return torch.cat(embeddings, dim=0)


def is_relevant(text, descriptor_embeddings, tokenizer, model, threshold=0.7):
    ''' 
        Verify the similarity between the specific data with the text. The method considers is the cosine_similarity.
        
        input: 
            text (str): Text base to encode and verify the similarity with topic. 
            descriptor_embeddings (list str): List of descriptors with specific focus of application. 
            tokenizer (transformers.BertTokenizer): The loaded Tokenizer to Bert LLM.
            model (transformers.BertForSequenceClassification): The loaded classification model. 
            batch_size (int): Size of batch applied in embedding.
        output:
            is_relevant (bool): If has any similarity, it is True, else is False.
    '''
    text_embedding = get_embeddings_batch([text], tokenizer, model, batch_size=1)

    similarities = [cosine_similarity(text_embedding, descr_embedding).item() for descr_embedding in descriptor_embeddings]

    return any(similarity >= threshold for similarity in similarities)


def initialize_weights(module):
    '''
        Initialize the weiths that will be applied in the pre-trained model. The method applied is xavier uniform

        input:
            module
    '''
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    print("The weights of model was initilized.")


def save_checkpoint(epoch, model, optimizer, filename):
    '''
        Create a file with the model and optimizer state exported in a checkpoint file.

        input:
            epoch (int): The current epoch of training.
            model (transformers.BertForSequenceClassification): The sentiment classification model.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            filename (str): The file path to save the checkpoint.
    '''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1} in the {filename}")


def save_onnx(onnx_model_path, model, input_ids, attention_mask): 
    '''
        To export the model and the state to onnx file. 

        input: 
            onnx_model_path (str): The file name of onnx that will be exported.
            model (transformers.BertForSequenceClassification): The sentiment classification model.
            input_ids (lst): List of inputs create to model. 
            attention_mask (lst): List of mask applied in the model. 
    '''
    onnx_model_path = "./model/bert_sequence_classification.onnx"

    # Exportar o modelo
    torch.onnx.export(
        model,                                
        (input_ids, attention_mask),          
        onnx_model_path,                      
        input_names=["input_ids", "attention_mask"],  
        output_names=["logits"],              
        dynamic_axes={                        
            "input_ids": {0: "batch_size"},   
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14                      #
    )

    print(f"Model exported to ONNX file sucessfully, path: {onnx_model_path}")


def extract_method_names(text, filename_descriptor="./DB/input/descriptors.json"):
    '''
        To extract the method name of each paper available in the dataframe.

        input: 
            text (lst str): Input text to extract the Techniques applied. 
        output: 
            methods_name (lst str): List of methods name of input text. 
    '''

    # Carregar o modelo de NER do spaCy
    nlp = spacy.load("en_core_web_sm")

    descriptors_json = preprocessing_data.load_descriptors_json(filename_descriptor)
    methods_names_list_df = []
    method_descriptors = []
    for i in ["deep learning", "nlp", "generative", "transformer", "multimodal", "vision", "machine learning", "techniques"]:
        method_descriptors.extend(descriptors_json[i])

    # Extrair nomes de métodos para cada paper
    for i, paper_text in enumerate( text ):
        # Processar o texto com spaCy para NER
        doc = nlp(paper_text)

        # Coletar entidades e descritores que aparecem no texto
        method_names = []

        # Usar o NER do spaCy para identificar entidades no texto
        for ent in doc.ents:
            if ent.label_ in ["MISC"]:
                method_names.append(ent.text)

        # Verificar se algum descritor específico aparece no texto
        for descriptor in method_descriptors:
            if descriptor in paper_text:
                method_names.append(descriptor)

        methods_names_list_df.append(method_names)
        
    return methods_names_list_df



def classify_semantic_methods(embeddings_relevant, relevant_papers ):
    '''
        input: 

        output: 
    '''

    n_pca=2
    pca = PCA(n_components=n_pca)
    embeddings_reduzidos = pca.fit_transform(embeddings_relevant)

    # Apply K-means to 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    rotulos = kmeans.fit_predict(embeddings_relevant)
    relevant_papers['cluster_label'] = rotulos

    silhouette_avg = silhouette_score(embeddings_relevant, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}") 

    cluster_to_label = {
        0 : "text mining",
        1 : "computer vision",
        2 : "computer vision and text mining",
        3 : "other"
    }
    # Atribuir rótulos semânticos aos textos com base nos clusters
    semantic_labels = [cluster_to_label[label] for label in rotulos]

    methods_names= []
    # Exibir os rótulos semânticos para cada texto
    for text, label in zip(relevant_papers['Concat Text'], semantic_labels):
        # print(f"Texto: {text} -> Rótulo Semântico: {label}")
        methods_names.append(label)

    plt.figure(figsize=(10, 6))
    cores = ['r', 'g', 'b', 'y']
    for i in range(2*n_pca):
        plt.scatter(embeddings_reduzidos[rotulos == i, 0], embeddings_reduzidos[rotulos == i, 1], c=cores[i], label=f'Cluster {i + 1}')
    plt.legend()
    plt.title("Clusters of methods")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("./fig_kmeans.png", format='png', dpi=300)

    return methods_names