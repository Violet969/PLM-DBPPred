import esm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from argparse import Namespace
import numpy as np



class ESM2Embedding(nn.Module):
    """ ESM1b embedding layer module """

    def __init__(self, embedding_args: dict, embedding_pretrained=None, ft_embed_tokens: bool = False, ft_transformer: bool = False, ft_contact_head: bool = False,
                 ft_embed_positions: bool = False, ft_emb_layer_norm_before: bool = True, ft_emb_layer_norm_after: bool = True, 
                 ft_lm_head: bool = True, max_embedding: int = 1024, offset: int = 200):
        """ Constructor
        Args:
            embedding_args: arguments to embeddings model
            embedding_pretrained: patht to pretrained model
            ft_embed_tokens: finetune embed tokens layer
            ft_transformer: finetune transformer layer
            ft_contact_head: finetune contact head
            ft_embed_positions: finetune embedding positions
            ft_emb_layer_norm_before: finetune embedding layer norm before
            ft_emb_layer_norm_after: finetune embedding layer norm after
            ft_lm_head: finetune lm head layer
            max_embeddings: maximum sequence length for language model
            offset: overlap offset when concatenating sequences above max embedding
        """
        super(ESM2Embedding, self).__init__()

        # if given model path then pretrain
        if embedding_pretrained:
            self.model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        else:
            # configure pre-trained model
            alphabet = esm.Alphabet.from_architecture(embedding_args['arch'])
            model_type = esm.ProteinBertModel

            self.model = model_type(Namespace(**embedding_args), alphabet,)

        self.max_embedding = max_embedding
        self.offset = offset

        # finetuning, freezes all layers by default
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
            ft_embed_positions, ft_emb_layer_norm_before, ft_emb_layer_norm_after, ft_lm_head]

        # finetune by freezing unchoosen layers
        for i, child in enumerate(self.model.children()):
            if self.finetune[i] == False:
                for param in child.parameters():
                    param.requires_grad = False
    

    def forward(self, batch_tokens: torch.tensor, padding_length: int = None) -> torch.tensor:
        """ Convert tokens to embeddings
        Args:
            batch_tokens: tensor with sequence tokens
        """
        batch_residues_original = batch_tokens.shape[1]

        # remove padding
        if padding_length:
            batch_tokens = batch_tokens[:, :padding_length]

        batch_residues = batch_tokens.shape[1]

        embedding = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[30])[
                               "representations"][30]

        batch_iter = math.ceil(batch_residues / (self.max_embedding - self.offset))

        # if size above 1024 then generate embeddings that overlaps with the offset
        if batch_residues >= self.max_embedding:
            # combine by overlaps
            for i in range(1, batch_iter):
                o1 = (self.max_embedding - self.offset) * i
                o2 = o1 + self.max_embedding

                if i == batch_iter - 1:
                    if o2 > batch_residues:
                        embedding = torch.cat([embedding[:, :o1], self.model(
                        batch_tokens[:, o1:batch_residues], repr_layers=[30])["representations"][30]], dim=1)
                    else:
                        embedding = torch.cat([embedding[:, :o2 - self.offset], self.model(
                        batch_tokens[:, o2 - self.offset:batch_residues], repr_layers=[30])["representations"][30]], dim=1)
                else:
                    embedding = torch.cat([embedding[:, :o1], self.model(
                        batch_tokens[:, o1:o2], repr_layers=[30])["representations"][30]], dim=1)


            embedding = torch.nan_to_num(embedding)

        # add padding
        if padding_length:
            embedding = F.pad(embedding, pad=(0, 0, 0, batch_residues_original
                            - padding_length), mode='constant', value=0)

        # cleanup
        del batch_tokens
        torch.cuda.empty_cache()
        # print(embedding[:, 1:embedding.shape[1]-1, :].shape)
        return embedding[:, 1:embedding.shape[1]-1, :]

class ESM2_only_sol(nn.Module):
    def __init__(self, init_n_channels: int, embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super(ESM2_only_sol, self).__init__()
        print('embedding_args',embedding_args)
        # ESM1b block
        self.embedding = ESM2Embedding(embedding_args, embedding_pretrained, **kwargs)

        feature_number=480
        self.global_avg_pool = nn.AdaptiveAvgPool2d((feature_number, 1)) # 全局平均池化层
        self.fc =  nn.Sequential(*[
            nn.Linear(in_features=feature_number, out_features=1),
            nn.Sigmoid()
        ])
        
    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """
        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))
        # print('x',x.shape)
        x = x.permute(0, 2, 1)

        x = self.global_avg_pool(x)
        # print('x',x.shape)
        x = x.view(x.size(0), -1) # 展平张量
        
        outputs = self.fc(x)
                

        return outputs
    
    
class ESM2_protlocal(nn.Module):
    def __init__(self, init_n_channels: int, output_dim: int, embeddings_dim: int, kernel_size: tuple, max_length: int, dropout: float,  conv_dropout: float,embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        
        super(ESM2_protlocal, self).__init__()
        
        # ESM2 block
        self.embedding = ESM2Embedding(embedding_args, embedding_pretrained, **kwargs)
        
        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 4)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 4)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Sequential(*[
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        ])
        

    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """
        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))
        
        x = x.permute(0, 2, 1)
        
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
       
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        
        o = self.linear(o) 
        outputs = self.output(o)
        return outputs