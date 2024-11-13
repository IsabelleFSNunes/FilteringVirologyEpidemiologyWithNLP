import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
        threshold (float): Value of threshould applied to similarity function.
    output:
        is_relevant (bool): If has any similarity, it is True, else is False.
    '''
    text_embedding = get_embeddings_batch([text], tokenizer, model, batch_size=1)

    similarities = [cosine_similarity(text_embedding, descr_embedding).item() for descr_embedding in descriptor_embeddings]

    return any(similarity >= threshold for similarity in similarities)


def initialize_weights(module):
    '''
        Initialize the weiths that will be applied in the pre-trained model. The method applied is xavier uniform.
    input:
        module (torch.nn.modules.linear.Linear): The last layer of classification, in the Linear format for BertForSequenceClassification. 
    '''
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    print("The weights of classification layer was initilized.")


def extract_method_names(text, filename_descriptor="./DB/input/descriptors.json"):
    '''
        To extract the method name of each paper available in the dataframe.
    input: 
        text (lst str): Input text to extract the Techniques applied. 
        filename_descriptor (str): path of filename descriptor in json format. 
    output: 
        methods_names_list_df (lst str): List of methods name of input text. 
    '''

    nlp = spacy.load("en_core_web_sm") # load the NER of spaCy

    descriptors_json = preprocessing_data.load_descriptors_json(filename_descriptor)
    methods_names_list_df = []
    method_descriptors = []

    for i in ["deep learning", "nlp", "generative", "transformer", "multimodal", "vision", "machine learning", "techniques"]:
        method_descriptors.extend(descriptors_json[i])


    for i, paper_text in enumerate( text ):
        doc = nlp(paper_text)
        method_names = []

        # Apply NER (SpaCy) to identify the entities of text input
        for entity in doc.ents:
            if entity.label_ in ["MISC"]:
                method_names.append(entity.text)

        # Verify the descriptors in the entities
        for descriptor in method_descriptors:
            if descriptor in paper_text:
                method_names.append(descriptor)

        methods_names_list_df.append(method_names)
        
    return methods_names_list_df



def classify_semantic_methods(embeddings_relevant, input_df, path_figure= "./fig_kmeans.png" ):
    '''
        Classify the texts from the embedding perspective of semantic model Bert. Export a figure with the four clusters obtained in the topic.
    input : 
        embeddings_relevant (torch.Tensor): Tensor considering the llm model applied the input text.
        input_df (pd.DataFrame): DataFrame of general database, considering the specific texts.
        path_figure (str) [optional]: It is a path of output image with the clusters in this execution. 
    output: 
        methods_names (list): A list with the all methods classified to input text 
    '''

    n_pca=2
    pca = PCA(n_components=n_pca)
    embeddings_reduzidos = pca.fit_transform(embeddings_relevant)

    # Apply K-means to 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    rotulos = kmeans.fit_predict(embeddings_relevant)
    input_df['cluster_label'] = rotulos

    silhouette_avg = silhouette_score(embeddings_relevant, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}") 

    cluster_to_label = {
        0 : "text mining",
        1 : "computer vision",
        2 : "computer vision and text mining",
        3 : "other"
    }
    # Apply the semantic labels to base text in clusters
    semantic_labels = [cluster_to_label[label] for label in rotulos]

    methods_names= []
    # Present the semantic label for each string on list
    for text, label in zip(input_df['Concat Text'], semantic_labels):
        methods_names.append(label)

    # Create the image 
    plt.figure(figsize=(10, 6))
    cores = ['r', 'g', 'b', 'y']
    for i in range(2*n_pca):
        plt.scatter(embeddings_reduzidos[rotulos == i, 0], embeddings_reduzidos[rotulos == i, 1], c=cores[i], label=f'Cluster {i + 1}')
    plt.legend()
    plt.title("Clusters of methods")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(path_figure, format='png', dpi=300)

    return methods_names