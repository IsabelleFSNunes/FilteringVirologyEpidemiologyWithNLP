# Filtering semantically with BERT 

## Introduction 

This application concentrate to specific topics about the medicine and technology, such as: Virology and Epidemiology, and the context of computational methods apply in this sector, Deep Learning and the branches of Natural Language and Computer Vision. 

### Objective

Create automatic filtering considering Deep Learning and techniques of NLP, to filter a dataset with a screening papers with advanced filtering by keywords with logic operators, as a query.

In this instance, the big task was partitioned in three tasks to create a semantic cascade of filtering approach.  

## Content and others details  

| Directory/File | Description |
| ---| ---| 
|**DB/** | Database used in this application - [README.md](./DB/README.md) |
|**model/** | Model (just this case, that is a short model, it was shared in github) | 
| **requirements.txt** | The requirements used to execution of this code.  |
|**preprocessing_data.py** | Methods considering the preprocess of data, such as adjust the inputs, load db and models. | 
| **processing_data.py** | Create the embedding data, initialize parameters, classify and extract methods. |
|**utils.py**| Methods to save and export file.|
|**main.py** | Execution of code | 

### Execution 
The code was created and executed in a conda environment. The hardware applied has the specs: 
```
Processor: i9-13900H
Graphical Card: RTX 4070
RAM Memory: 32 GB
Storage: 1TB SSD NVMe
```
The code execution, applying trained stage only for classification linear layer was approximatelly **15 min**.

```sh
pip install -r requirements.txt 

```

