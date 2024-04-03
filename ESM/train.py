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

import esm
from esm import Alphabet
from sklearn.model_selection import train_test_split 
import sklearn.metrics as metrics

from io import StringIO

torch.cuda.set_device(2)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5,6,7"
device = torch.device('cuda:2')   
device_ids=[2,3,4,5,6,7]

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

def data_loader(csv_data,batch_size):
    data = []
    for idx, row in csv_data.iterrows():
        data.append((row["protein_ID"],row["Sequence"]))
    labels = csv_data['label']
    embedding_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    #reverse sequences to tensor
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    print(batch_lens.shape)
    inputs = batch_tokens
    mask = batch_lens
    # reverse label to tensor
    label_dict = {label: i for i, label in enumerate(set(labels))}
    tokenized_labels = [label_dict[label] for label in labels]
    labels_tensor = torch.LongTensor(tokenized_labels)
    dataset = torch.utils.data.TensorDataset(inputs, mask, labels_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main(args):
    # seed_everything()
    torch.backends.cudnn.benchmark = True
    ####load model
   
    model = ESM2_protlocal(**MODEL_CONFIG)
    print(model)
    print('model_parameter_number',get_parameter_number(model))
    
    model = torch.nn.DataParallel(model, device_ids=device_ids,dim=0)
    model.to(device)
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    ###### load_data
    all_train_data = pd.read_csv(args.train_set)

    # split train set and test set
    train_data, val_data = train_test_split(all_train_data, test_size=0.05, random_state=1234)
    test_data = pd.read_csv(args.test_set)
    
    train_data_loader = data_loader(train_data,args.batch_size)
    val_data_loader = data_loader(val_data,args.batch_size)
    test_data_loader = data_loader(test_data,args.batch_size)
    io = IOStream('outputs/' + args.exp_name + '/run.log')
    ### train
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for batch in train_data_loader:
            batch_inputs = batch[0].to(device)
           
            mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs,mask)
            labels = torch.tensor(batch_labels, dtype=torch.long).float()
            print('outputs',outputs.squeeze(1))
            print('batch_labels',batch_labels)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels)
            print('loss',loss)
            loss.backward()
            optimizer.step()
            
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(labels.cpu().numpy())
        
            train_pred.append((outputs.detach().cpu().numpy() >= 0.5))
            
        

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        #Val
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for batch in val_data_loader:
            batch_inputs = batch[0].to(device)
            mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            outputs = model(batch_inputs,mask)
            labels = torch.tensor(batch_labels, dtype=torch.long).float()
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels)
            lr_scheduler.step(loss)
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(labels.cpu().numpy())
            test_pred.append((outputs.detach().cpu().numpy() >= 0.5))
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'outputs/{exp}/models/model-{epoch}.t7'.format(exp=args.exp_name,epoch=str(epoch)))
            print("save weights!!!")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_set", help="Train file path", required=True)
    parser.add_argument("-test_set", help="Test file path", required=True)
    parser.add_argument("-gpu", help="Set to use GPU", default = True)
    parser.add_argument("-batch_size", help="Batch size", default = 119)
    parser.add_argument("-learning_rate", help="learning rate", default = 0.002)
    parser.add_argument("-num_epochs", help="number of epochs", default = 15)
    parser.add_argument('-exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    
    args = parser.parse_args()
    _init_()
    
    io = IOStream('outputs/'+args.exp_name + '/run.log')
    io.cprint(str(args))
    
    tempdir = 'outputs/'+args.exp_name+'/'
    os.makedirs(tempdir, exist_ok=True)

    main(args)
                           
