import torch
from torch.utils.data import Dataset

class VirologyPapersDataset(Dataset):
    """
        TODO: TO COMPLETE THE DESCRIPTION
        Class to create a object to Virology Paper
        inputs: 

        output: 
    """
    def __init__( self, dataframe, tokenizer, max_length ):
        self.length= len( dataframe )
        self.data= dataframe
        self.tokenizer= tokenizer
        self.max_length= max_length

    def __getitem__( self, index ):
        text= str( self.data["Concat Text"].iloc[index] )
        text= " ".join( text.split() )
        inputs= self.tokenizer.encode_plus( text,
                                            None,
                                            add_special_tokens= True,
                                            max_length= self.max_length,
                                            padding= 'max_length',
                                            return_token_type_ids= True,
                                            truncation= True )
        ids= inputs['input_ids']
        mask= inputs['attention_mask']
        token_type_ids= inputs["token_type_ids"]

        return {'input_ids': torch.tensor(ids, dtype= torch.long),
                'attention_mask': torch.tensor(mask, dtype= torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype= torch.long) }

    def __len__( self ):
        return self.length
