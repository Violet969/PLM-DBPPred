# The examples in this notebook use a set of nine benchmarks described in our publication.
# These benchmarks can be downloaded via FTP from: ftp.cs.huji.ac.il/users/nadavb/protein_bert/protein_benchmarks
# Download the benchmarks into a directory on your machine and set the following variable to the path of that directory.

import os
from io import StringIO
import argparse
import pandas as pd
from IPython.display import display
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        
def main(args):
    # A local (non-global) bianry output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)


    # Loading the dataset

    train_set_file_path = os.path.join(args.train_set)
    train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
    train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.04, random_state = 0)

    test_set_file_path = os.path.join(args.test_set)
    test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

    # print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

    io.cprint(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')
    # Loading the pre-trained model and fine-tuning it on the loaded dataset

    pretrained_model_generator, input_encoder = load_pretrained_model()

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
            get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ]

    finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
            seq_len = 256, batch_size = args.batch_size, max_epochs_per_stage =args.num_epochs, lr = args.learning_rate, begin_with_frozen_pretrained_layers = True, \
            lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)

    X_test,y_preds,y_trues,results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['protein_ID'],test_set['Sequence'], test_set['label'], \
            start_seq_len = 512, start_batch_size = 32)

    
    dataframe_512 = pd.DataFrame()
    dataframe_1024 = pd.DataFrame()
    dataframe_2048 = pd.DataFrame()


    dataframe_512['protein_ID'] = X_test[0]
    dataframe_512['y_preds'] = y_preds[0]
    dataframe_512['y_trues'] = y_trues[0]
    dataframe_1024['protein_ID'] = X_test[1]
    dataframe_1024['y_preds'] = y_preds[1]
    dataframe_1024['y_trues'] = y_trues[1]
    dataframe_2048['protein_ID'] = X_test[2]
    dataframe_2048['y_preds'] = y_preds[2]
    dataframe_2048['y_trues'] = y_trues[2]

    dataframe = pd.concat([dataframe_512,dataframe_1024,dataframe_2048])

    dataframe.to_csv(args.exp_name+'PLM-DBPPred_predict_result.csv')
    
    with open('outputs/'+args.exp_name+'/models/model.pt', 'wb') as f:
        pickle.dump(model_generator, f)

    with open('outputs/'+args.exp_name+'/models/model_OUTPUT_SPEC.pt', 'wb') as f:
        pickle.dump(OUTPUT_SPEC, f)

    with open('outputs/'+args.exp_name+'/models/model_input_encoder.pt', 'wb') as f:
        pickle.dump(input_encoder, f)
    # Evaluating the performance on the test-set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_set", help="Train file path", required=True)
    parser.add_argument("-test_set", help="Test file path", required=True)
    parser.add_argument("-batch_size", help="Batch size", default = 32)
    parser.add_argument("-learning_rate", help="learning rate", default = 1e-04)
    parser.add_argument("-num_epochs", help="number of epochs", default = 10)
    parser.add_argument('-exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    
    args = parser.parse_args()
    _init_()
    
    io = IOStream('outputs/'+args.exp_name + '/run.log')
    io.cprint(str(args))
    
    tempdir = 'outputs/'+args.exp_name+'/'
    os.makedirs(tempdir, exist_ok=True)

    main(args)
