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
import nsp3
from nsp3.models import ESM2_protlocal
from nsp3.config_protlocal import *
from nsp3.utils import IOStream,seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

import esm
from esm import Alphabet
from sklearn.model_selection import train_test_split 
import sklearn.metrics as metrics

from io import StringIO


class CustomDataset(Dataset):
    def __init__(self, ids, sequences,embeddings,lengths,labels):
        self.ids = ids
        self.sequences = sequences
        self.embeddings = embeddings
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        data_id = self.ids[index]
        sequence = self._encode_sequence(self.sequences[index])  # Encode the string sequence to a tensor
        embeddings = self.embeddings[index]
        lengths = self.lengths[index]
        label = self.labels[index]
        return data_id, sequence,embeddings,lengths,label

    def _encode_sequence(self, sequence):
        # This is just an example; you may use more sophisticated encoding schemes here
        # For simplicity, let's convert each character to its ASCII code
        return torch.tensor([ord(char) for char in sequence])

    
def data_loader(csv_data,batch_size):
    data = []
    for idx, row in csv_data.iterrows():
        data.append((row["protein_ID"],row["Sequence"]))
    labels = csv_data['label']
    embedding_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    batch_id, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    print(batch_lens.shape)
   
    label_dict = {label: i for i, label in enumerate(set(labels))}
    tokenized_labels = [label_dict[label] for label in labels]
    labels_tensor = torch.LongTensor(tokenized_labels)
   
    dataset = CustomDataset(batch_id,batch_strs,batch_tokens,batch_lens,labels_tensor)
    # torch.utils.data.TensorDataset(torch.Tensor(batch_id),batch_tokens,batch_lens,labels_tensor)
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
        batch_labels = batch[4].to(device)
        outputs = model(batch_inputs,mask)
        labels = torch.tensor(batch_labels, dtype=torch.long).float()
        
        count += batch_size
        identifiers.append(batch_id)
        sequences.append(batch_sequences)
        test_true.append(labels.cpu().numpy())
        test_pred.append((outputs.detach().cpu().numpy() >= 0.5))
        test_pred_score.append(outputs.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'test acc: %.6f, test avg acc: %.6f' % (test_acc,avg_per_class_acc)
    print(outstr)
    prediction_result = pd.DataFrame(columns=['protein_ID','predict_result'])
    
    prediction_result['protein_ID'] = np.array(identifiers).squeeze(1).tolist()
    
    prediction_result['predict_result'] = np.array(test_pred_score).squeeze(1).tolist()
    prediction_result.to_csv(args.o+'prediction_result.csv')


    
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
