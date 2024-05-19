import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
import time
import pandas as pd
from zipfile import ZipFile
import torch.nn.functional as F
import model_code
from model_code.models import ESM2_protlocal
from model_code.config_protlocal import *
from model_code.utils import IOStream,seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

import esm
from esm import Alphabet
from sklearn.model_selection import train_test_split 
import sklearn.metrics as metrics

from io import StringIO


class CustomDataset(Dataset):
    def __init__(self, ids, sequences,embeddings,lengths):
        self.ids = ids
        self.sequences = sequences
        self.embeddings = embeddings
        self.lengths = lengths
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        data_id = self.ids[index]
        sequence = self._encode_sequence(self.sequences[index])  # Encode the string sequence to a tensor
        embeddings = self.embeddings[index]
        lengths = self.lengths[index]
        
        return data_id, sequence,embeddings,lengths

    def _encode_sequence(self, sequence):
        
        return torch.tensor([ord(char) for char in sequence])

    
def data_loader(csv_data,batch_size):
    data = []
    for idx, row in csv_data.iterrows():
        data.append((row["protein_ID"],row["Sequence"]))
    
    embedding_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    
    batch_converter = alphabet.get_batch_converter()
    batch_id, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    print(batch_lens.shape)
   
    
   
    dataset = CustomDataset(batch_id,batch_strs,batch_tokens,batch_lens)
   
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    ####load model
   
    model = ESM2_protlocal(**MODEL_CONFIG)
    model_data = torch.load(args.m, map_location = device)
    model_data = {k.replace("module.", ""): v for k, v in model_data.items()}
    model.load_state_dict(model_data)
    model.eval()
    
    print(model)

    model.to(device)
    # print([{'params': model.parameters(), **MODEL_CONFIG}])

    batch_size = args.batch_size
 
    test_data = pd.read_csv(args.test_set)
    
   
    test_data_loader = data_loader(test_data,args.batch_size)
    
    ####################
    #Test
    ####################
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    identifiers =[]
    sequences = []
    test_pred_score = []
    for batch in test_data_loader:
        # print(batch)
        batch_id = batch[0]
        batch_sequences = batch[1]
        batch_inputs = batch[2].to(device)
        mask = batch[3].to(device)
        
        outputs = model(batch_inputs,mask)
        
        
        count += batch_size
        identifiers.append(batch_id)
        
        test_pred_score.append(outputs.detach().cpu().numpy())
    
    prediction_result = pd.DataFrame(columns=['protein_ID','predict_result'])
    
    prediction_result['protein_ID'] = np.array(identifiers).squeeze(1).tolist()
    model_output = np.array(test_pred_score).squeeze(1).tolist()
    prediction_result['predict_result'] = [a for a in model_output]
    prediction_result['predict_result'] = prediction_result['predict_result'].astype("string")
    prediction_result['predict_result'] = prediction_result['predict_result'].str.replace('[', '')
    prediction_result['predict_result'] = prediction_result['predict_result'].str.replace(']', '')
    prediction_result.to_csv(args.o+'ESM_prediction_result.csv',index=False)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-test_set", help="File path", required=True)
    parser.add_argument("-o", help="File output path", required=True)
    parser.add_argument("-m", help="Model data path", default = "./ESM2_30_model_param/model/model-7.t7")
    parser.add_argument("-gpu", help="Set to use GPU", default = True)
    parser.add_argument("-batch_size", help="Batch size", default = 1)
    args = parser.parse_args()
    
    tempdir = args.o+'/'
    os.makedirs(tempdir, exist_ok=True)
    main(args)
    
