import torch
import onnx 

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

    # To export the model
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
        opset_version=14
    )

    print(f"Model exported to ONNX file sucessfully, path: {onnx_model_path}")
