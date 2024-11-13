# Filtering semantically with BERT 

## Introduction 

This application concentrate to specific topics about the medicine and technology, such as: Virology and Epidemiology, and the context of computational methods apply in this sector, Deep Learning and the branches of Natural Language and Computer Vision. 

### Objective

Create automatic filtering considering Deep Learning and techniques of NLP, to filter a dataset with a screening papers with advanced filtering by keywords with logic operators, as a query.


## Methodology

In this instance, the big task was partitioned in three tasks to create a semantic cascade of filtering approach.

Starting with the complete database, the relevant subgroup of papers was selected applying the semantic of Bidirectional Encoder Representations for Transformers (BERT), considering the cosine of similarity between the embedding created by [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) pre-trained model and the descriptors initialized manually with the content of keywords of the creation of database. 

With the relevant papers filtering with the BERT model, the database decreased and passed to another stage. To apply classification of the database, the PCA was applied to summary the dimensions of vectorial space of embeddings. To continue the analysis and classification the Kmeans is considered to clusterizing the topics of embedding with more similarities. Emphasizing fours classes listed, ["text mining", "computer vision", "both", "other"]. To simplify, the label "both" was changed as a concatenation between the two first labels, "text mining and computer vision".

Currently, the new attributes already inserted in the dataframe, the last aim is extract the central topic and technique include in each relevant paper. For that, the descriptor of techniques in deep learning is conectaded with entities created by function of NER pipeline.

## Materials and Details
Applying the methodology, the respective files was considered organized as shown in Table 1.


<center>Table 1: Contents of repository and descriptions.</center>

| Directory/File | Description |
| ---| ---| 
|**DB/** | Database used in this application - [README.md](./DB/README.md) |
|**model/** | Model (just this case, that is a short model, it was shared in github) | 
| **requirements.txt** | The requirements used to execution of this code.  |
|**preprocessing_data.py** | Methods considering the preprocess of data, such as adjust the inputs, load db and models. | 
| **processing_data.py** | Create the embedding data, initialize parameters, classify and extract methods. |
|**utils.py**| Methods to save and export file.|
|**main.py** | Execution of code | 


### Environment
The code was created and executed in a conda environment. The hardware applied has the specs: 
```
Processor: i9-13900H
Graphical Card: RTX 4070
RAM Memory: 32 GB
Storage: 1TB SSD NVMe
```

### Execution

The packages used in this repository, such Pytorch, Pandas, Numpy, Onnx, are listed in the requirements file, and could be updated by the instruction on the terminal. 

```sh
pip install -r requirements.txt 

```

The execution just need the python 3.10 pre-installed:
```
python main.py
```

The code execution, applying trained stage only for classification linear layer in this machine was approximatelly **15 min**.




## Conclusion