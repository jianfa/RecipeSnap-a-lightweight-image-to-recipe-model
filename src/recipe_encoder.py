# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer

    Parameters
    ----------
    dropout : float
        Dropout value.
    num_embeddings : int
        Number of embeddings to train.
    hidden_dim : int
        Embedding dimensionality
    """

    def __init__(self, dropout=0.1, num_embeddings=50, hidden_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)


def AvgPoolSequence(attn_mask, feats, e=1e-12):
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats


class SingleTransformerEncoder(nn.Module):
    """A transformer encoder with masked average pooling at the output

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, dim, n_heads, n_layers):
        super(SingleTransformerEncoder, self).__init__()

        self.pos_encoder = LearnedPositionalEncoding(hidden_dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads)

        self.tf = nn.TransformerEncoder(encoder_layer,
                                        num_layers=n_layers)

    def forward(self, feat, ignore_mask):

        if self.pos_encoder is not None:
            feat = self.pos_encoder(feat)
        # reshape input to t x bs x d
        feat = feat.permute(1, 0, 2)
        out = self.tf(feat, src_key_padding_mask=ignore_mask)
        # reshape back to bs x t x d
        out = out.permute(1, 0, 2)

        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)

        return out


class RecipeTransformerEncoder(nn.Module):
    """The recipe text encoder. Encapsulates encoders for all recipe components.

    Parameters
    ----------
    vocab_size : int
        Input size (recipe vocabulary).
    hidden_size : int
        Output embedding size.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, vocab_size, hidden_size, n_heads,
                 n_layers):
        super(RecipeTransformerEncoder, self).__init__()

        # Embedding layer mapping vocabulary words to embeddings.
        # The embedding layer is common for all text tokens (ingrs, instrs, title)
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)

        self.tfs = nn.ModuleDict()

        # independent transformer encoder for each recipe component
        for name in ['title', 'ingredients', 'instructions']:
            self.tfs[name] = SingleTransformerEncoder(dim=hidden_size,
                                                      n_heads=n_heads,
                                                      n_layers=n_layers
                                                     )

        # second transformer for sequences of sequences inputs
        # (eg a list of raw ingredients or a list of instructions)
        self.merger = nn.ModuleDict()
        for name in ['ingredients', 'instructions']:
            self.merger[name] = SingleTransformerEncoder(dim=hidden_size,
                                                         n_heads=n_heads,
                                                         n_layers=n_layers)

    def forward(self, input, name=None):
        '''
        Extracts features for an input using the corresponding encoder (by name)
        '''
        # check if input is a sequence or a sequence of sequences
        if len(input.size()) == 2:
            # if it is a sequence, the output of a single transformer is used
            ignore_mask = (input == 0)
            out = self.tfs[name](self.word_embedding(input), ignore_mask)

        else:
            # if it's a sequence of sequences, the first encoder is applied
            # to each sentence, and the second on

            # reshape from BxNxTxD to BNxTxD
            input_rs = input.view(input.size(0)*input.size(1), input.size(2))
            ignore_mask = (input_rs == 0)

            # trick to avoid nan behavior with fully padded sentences
            # (due to batching)
            ignore_mask[:, 0] = 0
            out = self.tfs[name](self.word_embedding(input_rs), ignore_mask)

            # reshape back
            out = out.view(input.size(0), input.size(1), out.size(-1))

            # create mask for second transformer
            attn_mask = input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()

            out = self.merger[name](out, torch.logical_not(mask_list))

        return out

class JointEmbedding(nn.Module):
    """ Extract the join embedding of recipes title, ingredient, and instructions.
    Parameters
    ----------
    output_size : int
        Embedding output size.
    vocab_size : int
        Input size for recipes.
    hidden_recipe : int
        Embedding size for recipe components
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    """

    def __init__(self, output_size, 
                 vocab_size=16303,
                 hidden_recipe=512,
                 n_heads=4, n_layers=2):
        super(JointEmbedding, self).__init__()

        self.text_encoder = RecipeTransformerEncoder(vocab_size,
                                                     hidden_size=hidden_recipe,
                                                     n_heads=n_heads,
                                                     n_layers=n_layers)

        self.merger_recipe = nn.ModuleList()

        # linear layer to merge features from all recipe components.
        self.merger_recipe = nn.Linear(hidden_recipe*(3), output_size)

    def forward(self, title, ingrs, instrs,
                freeze_backbone=True, resume_recipe_encoder_only=False):

        text_features = []
        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}

        names = list(elems.keys())

        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
            text_feature = self.text_encoder(input_source, name)
            text_features.append(text_feature)

        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))

        recipe_feat = nn.Tanh()(recipe_feat)

        return recipe_feat