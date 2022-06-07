"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my youtube channel!


"""

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.randn(1, embed_size)) #nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape[1], x.shape[0]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(x + self.position_embedding) # TODO: try x + self.position_embedding

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=1088,
        num_layers=5,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="gpu",
        max_length=20,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.max_length = max_length
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.out = nn.Linear(embed_size, trg_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src == self.src_pad_idx).unsqueeze(1)
        # (N, 1, 1, src_len)
        return None # src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        """trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )"""
        #return trg_mask.to(self.device)

        nopeak_mask = torch.triu(torch.ones((1, trg_len, trg_len)), diagonal=1)
        nopeak_mask = Variable(nopeak_mask == 0)
        #print((trg != 0).unsqueeze(1).to(self.device) & nopeak_mask.to(self.device))
        #print(" -->", ((trg != 0).unsqueeze(1).to(self.device) & nopeak_mask.to(self.device)).shape)
        return ((trg != 0).unsqueeze(1).to(self.device) & nopeak_mask.to(self.device)).unsqueeze(1)

    def make_pad_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        pad_mask = torch.tril(torch.zeros((size, size))).expand(32, 1, size, size)
        return pad_mask.to(self.device)  # pad_mask.to

    def predict(self, src, max_length=20, SOS_token=0, EOS_token=3):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        """self.eval()
        final_out = []
        for i in range(src.shape[0]):
            src_aux = src[i, :, :]
            src_mask = self.make_src_mask(src_aux)
            enc_src = self.encoder(src_aux, src_mask)

            y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=self.device)

            for _ in range(max_length):
                # Get source mask
                trg_mask = self.make_pad_mask(y_input.size(1)).to(self.device)

                pred = self.decoder(y_input, enc_src, src_mask, trg_mask)

                next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
                next_item = torch.tensor([[next_item]], device=device)

                # Concatenate previous input with predicted best word
                y_input = torch.cat((y_input, next_item), dim=1)

                # Stop if model predicts end of sentence
                if next_item.view(-1).item() == EOS_token:
                    break

            final_out.append(y_input.view(-1).tolist())
        return final_out"""
        self.eval()
        src = torch.transpose(src, 0, 1)
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        outputs = torch.zeros(src.shape[0], max_length)
        outputs[:, 0] = 1
        for i in range(1, max_length):
            trg_mask = torch.triu(torch.ones((1, i, i)), diagonal=1)
            trg_mask = Variable(trg_mask == 0).cuda()
            out = self.decoder(outputs[:,:i].int().cuda(),
                                          enc_src, src_mask, trg_mask)
            out = F.softmax(out, dim=-1)
            val, ix = out[:, -1].data.topk(1)
            outputs[:, i] = torch.squeeze(ix)
            #if ix[0][0] == 3:
            #    break

        return out

    def forward(self, src, trg):
        if trg is None:
            #trg = Variable(torch.LongTensor(
            #    [self.trg_pad_idx] * src.shape[0]*self.max_length)).view(src.shape[0], self.max_length).to(self.device)
            #trg[:, 0] = 1
            #trg_mask = self.make_pad_mask(self.max_length, src.shape[0])
            return self.predict(src)

        src = torch.transpose(src, 0, 1) # (N, 1, dim_embedding)

        trg_mask = self.make_trg_mask(trg)

        #print(" trg -->", trg)
        src_mask = self.make_src_mask(src)
        #print(" mask -->", trg_mask)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        # out = F.log_softmax(self.out(enc_src), dim=-1)

        return out





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
