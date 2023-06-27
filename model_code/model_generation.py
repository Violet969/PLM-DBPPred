from numbers import Number
import pickle

import numpy as np

from tensorflow import keras

from .shared_utils.util import log
from .tokenization import additional_token_to_index, n_tokens, tokenize_seq

####
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K
import tensorflow as tf


class ModelGenerator:

    def __init__(self, optimizer_class = keras.optimizers.Adam, lr = 2e-04, other_optimizer_kwargs = {}, model_weights = None, optimizer_weights = None):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.other_optimizer_kwargs = other_optimizer_kwargs
        self.model_weights = model_weights
        self.optimizer_weights = optimizer_weights
        
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, lr = None, callbacks = [], **create_model_kwargs):
    
        train_X, train_Y, train_sample_weigths = encoded_train_set
        self.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
        model = self.create_model(seq_len, **create_model_kwargs)
        
        if lr is not None:
            model.optimizer.lr = lr
        
        model.fit(train_X, train_Y, sample_weight = train_sample_weigths, batch_size = batch_size, epochs = n_epochs, validation_data = encoded_valid_set, \
                callbacks = callbacks)
        self.update_state(model)
        
    def update_state(self, model):
        self.model_weights = copy_weights(model.get_weights())
        self.optimizer_weights = copy_weights(model.optimizer.get_weights())
        
    def _init_weights(self, model):
    
        if self.optimizer_weights is not None:
            # For some reason keras requires this strange little hack in order to properly initialize a new model's optimizer, so that
            # the optimizer's weights can be reloaded from an existing state.
            self._train_for_a_dummy_epoch(model)
            
        if self.model_weights is not None:
            model.set_weights(copy_weights(self.model_weights))
        
        if self.optimizer_weights is not None:
            if len(self.optimizer_weights) == len(model.optimizer.get_weights()):
                model.optimizer.set_weights(copy_weights(self.optimizer_weights))
            else:
                log('Incompatible number of optimizer weights - will not initialize them.')
            
    def _train_for_a_dummy_epoch(self, model):
        X, Y = self.dummy_epoch
        model.fit(X, Y, batch_size = 1, verbose = 0)
        
class PretrainingModelGenerator(ModelGenerator):

    def __init__(self, create_model_function, n_annotations, create_model_kwargs = {}, optimizer_class = keras.optimizers.Adam, lr = 2e-04, other_optimizer_kwargs = {}, \
            annots_loss_weight = 1, model_weights = None, optimizer_weights = None):
        
        ModelGenerator.__init__(self, optimizer_class = optimizer_class, lr = lr, other_optimizer_kwargs = other_optimizer_kwargs, model_weights = model_weights, \
                optimizer_weights = optimizer_weights)
        
        self.create_model_function = create_model_function
        self.n_annotations = n_annotations
        self.create_model_kwargs = create_model_kwargs
        self.annots_loss_weight = annots_loss_weight
        
    def create_model(self, seq_len, compile = True, init_weights = True):
        
        clear_session()
        model = self.create_model_function(seq_len, n_tokens, self.n_annotations, **self.create_model_kwargs)
        
        if compile:
            model.compile(optimizer =self.optimizer_class(learning_rate = self.lr, **self.other_optimizer_kwargs), loss = ['sparse_categorical_crossentropy', 'binary_crossentropy'], \
                    loss_weights = [1, self.annots_loss_weight])
        
        if init_weights:
            self._init_weights(model)
        
        return model
    
    
####Attention
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)
        
class FinetuningModelGenerator(ModelGenerator):

    def __init__(self, pretraining_model_generator, output_spec, pretraining_model_manipulation_function = None, dropout_rate = 0.5, optimizer_class = None, \
            lr = None, other_optimizer_kwargs = None, model_weights = None, optimizer_weights = None):
        
        if other_optimizer_kwargs is None:
            if optimizer_class is None:
                other_optimizer_kwargs = pretraining_model_generator.other_optimizer_kwargs
            else:
                other_optimizer_kwargs = {}
        
        if optimizer_class is None:
            optimizer_class = pretraining_model_generator.optimizer_class
            
        if lr is None:
            lr = pretraining_model_generator.lr
            
        ModelGenerator.__init__(self, optimizer_class = optimizer_class, lr = lr, other_optimizer_kwargs = other_optimizer_kwargs, model_weights = model_weights, \
                optimizer_weights = optimizer_weights)
        
        self.pretraining_model_generator = pretraining_model_generator
        self.output_spec = output_spec
        self.pretraining_model_manipulation_function = pretraining_model_manipulation_function
        self.dropout_rate = dropout_rate
                    
    def create_model(self, seq_len, freeze_pretrained_layers = False):
        
        model = self.pretraining_model_generator.create_model(seq_len, compile = False, init_weights = (self.model_weights is None))
            
        if self.pretraining_model_manipulation_function is not None:
            model = self.pretraining_model_manipulation_function(model)
            
        if freeze_pretrained_layers:
            for layer in model.layers:
                layer.trainable = False
        
        model_inputs = model.input
        pretraining_output_seq_layer, pretraining_output_annoatations_layer = model.output
        last_hidden_layer = pretraining_output_seq_layer if self.output_spec.output_type.is_seq else pretraining_output_annoatations_layer
        
        
        last_hidden_layer = keras.layers.Dropout(self.dropout_rate)(last_hidden_layer)
        
        if self.output_spec.output_type.is_categorical:
            output_layer = keras.layers.Dense(len(self.output_spec.unique_labels), activation = 'softmax')(last_hidden_layer)
            loss = 'sparse_categorical_crossentropy'
        ##change add attention
        elif self.output_spec.output_type.is_binary:
            # print('last_hidden_layer',last_hidden_layer.shape)
            #last_hidden_layer (None, 15599)
            last_hidden_layer = tf.expand_dims(last_hidden_layer,1)
            # print('last_hidden_layer',last_hidden_layer.shape)
            attention_layer = AttentionLayer()(last_hidden_layer)
            mlp_layer = keras.layers.Dense(2048, activation = 'sigmoid')(attention_layer)
            mlp_layer = tf.expand_dims(mlp_layer,1)
            attention_layer_2 = AttentionLayer()(mlp_layer)
            # print('attention_layer ',attention_layer_2.shape)
            output_layer = keras.layers.Dense(1, activation = 'sigmoid')(attention_layer_2)
            loss = 'binary_crossentropy'
        elif self.output_spec.output_type.is_numeric:
            output_layer = keras.layers.Dense(1, activation = None)(last_hidden_layer)
            loss = 'mse'
        else:
            raise ValueError('Unexpected global output type: %s' % self.output_spec.output_type)
                
        model = keras.models.Model(inputs = model_inputs, outputs = output_layer)
        model.compile(loss = loss, optimizer =self.optimizer_class(learning_rate = self.lr, **self.other_optimizer_kwargs))
        
        self._init_weights(model)
                
        return model
                        
class InputEncoder:

    def __init__(self, n_annotations):
        self.n_annotations = n_annotations

    def encode_X(self, seqs, seq_len):
        return [
            tokenize_seqs(seqs, seq_len),
            np.zeros((len(seqs), self.n_annotations), dtype = np.int8)
        ]
        
def load_pretrained_model_from_dump(dump_file_path, create_model_function, create_model_kwargs = {}, optimizer_class = keras.optimizers.Adam, lr = 2e-04, \
        other_optimizer_kwargs = {}, annots_loss_weight = 1, load_optimizer_weights = False):
    
    with open(dump_file_path, 'rb') as f:
        n_annotations, model_weights, optimizer_weights = pickle.load(f)
        
    if not load_optimizer_weights:
        optimizer_weights = None
    
    model_generator = PretrainingModelGenerator(create_model_function, n_annotations, create_model_kwargs = create_model_kwargs, optimizer_class = optimizer_class, lr = lr, \
            other_optimizer_kwargs = other_optimizer_kwargs, annots_loss_weight = annots_loss_weight, model_weights = model_weights, optimizer_weights = optimizer_weights)
    input_encoder = InputEncoder(n_annotations)
    
    return model_generator, input_encoder

def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)
    
def clear_session():
    import tensorflow.keras.backend as K
    K.clear_session()
    
def copy_weights(weights):
    return [_copy_number_or_array(w) for w in weights]
    
def _copy_number_or_array(variable):
    if isinstance(variable, np.ndarray):
        return variable.copy()
    elif isinstance(variable, Number):
        return variable
    else:
        raise TypeError('Unexpected type %s' % type(variable))
    
def _slice_arrays(arrays, slicing):
    if isinstance(arrays, list) or isinstance(arrays, tuple):
        return [array[slicing] for array in arrays]
    else:
        return arrays[slicing]
