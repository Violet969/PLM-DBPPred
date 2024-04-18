# The examples in this notebook use a set of nine benchmarks described in our publication.
# These benchmarks can be downloaded via FTP from: ftp.cs.huji.ac.il/users/nadavb/protein_bert/protein_benchmarks
# Download the benchmarks into a directory on your machine and set the following variable to the path of that directory.

import os
import argparse
import pandas as pd
from IPython.display import display
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len_prediction
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import pickle

def main(args):
    with open(args.m+'/model_attention.pt', 'rb') as f:
        model_generator = pickle.load(f)
    with open(args.m+'/model_input_encoder_attention.pt', 'rb') as f:
        input_encoder = pickle.load(f)
    with open(args.m+'/model_OUTPUT_SPEC_attention.pt', 'rb') as f:
        OUTPUT_SPEC = pickle.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_set = pd.read_csv(args.test_set).dropna().drop_duplicates()

    X_test,y_preds = evaluate_by_len_prediction(model_generator, input_encoder, OUTPUT_SPEC, test_set['protein_ID'],test_set['Sequence'], \
            start_seq_len = 512, start_batch_size = 1)

    dataframe_512 = pd.DataFrame()
    dataframe_1024 = pd.DataFrame()
    dataframe_2048 = pd.DataFrame()
    dataframe_4096 = pd.DataFrame()
    dataframe_8192 = pd.DataFrame()
    dataframe_16384 = pd.DataFrame()
    dataframe_32768 = pd.DataFrame()
    dataframe_65536 = pd.DataFrame()

    dataframe_512['protein_ID'] = X_test[0]
    dataframe_512['y_preds'] = y_preds[0]

    dataframe_1024['protein_ID'] = X_test[1]
    dataframe_1024['y_preds'] = y_preds[1]

    dataframe_2048['protein_ID'] = X_test[2]
    dataframe_2048['y_preds'] = y_preds[2]


    dataframe = pd.concat([dataframe_512,dataframe_1024,dataframe_2048])

    dataframe.to_csv(args.o+'DBP_proteinBERT.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-test_set", help="File path", required=True)
    parser.add_argument("-o", help="File output path", required=True)
    parser.add_argument("-m", help="Model data path", default = True)
    parser.add_argument("-gpu", help="Set to use GPU", default = True)
    parser.add_argument("-batch_size", help="Batch size", default = 1)
    args = parser.parse_args()
    
    tempdir = args.o+'/'
    os.makedirs(tempdir, exist_ok=True)
    main(args)
