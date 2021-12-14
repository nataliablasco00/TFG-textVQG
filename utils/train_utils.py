"""Utility functions for training.
"""

import json

import fasttext
import torch
import torchtext
import smart_open
from gensim.models.fasttext import FastText
#from gensim.models.wrappers import FastText


class Vocabulary(object):
    """Keeps track of all the words in the vocabulary.
    """

    # Reserved symbols
    SYM_PAD = '<pad>'    # padding.
    SYM_SOQ = '<start>'  # Start of question.
    SYM_SOR = '<resp>'   # Start of response.
    SYM_EOS = '<end>'    # End of sentence.
    SYM_UNK = '<unk>'    # Unknown word.

    def __init__(self):
        """Constructor for Vocabulary.
        """
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_PAD)
        self.add_word(self.SYM_SOQ)
        self.add_word(self.SYM_SOR)
        self.add_word(self.SYM_EOS)
        self.add_word(self.SYM_UNK)

    def add_word(self, word):
        """Adds a new word and updates the total number of unique words.
        Args:
            word: String representation of the word.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def remove_word(self, word):
        """Removes a specified word and updates the total number of unique words.
        Args:
            word: String representation of the word.
        """
        if word in self.word2idx:
            self.word2idx.pop(word)
            self.idx2word.pop(self.idx)
            self.idx -= 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'w') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    def load(self, location):
        with open(location, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']

    def tokens_to_words(self, tokens):
        """Converts tokens to vocab words.
        Args:
            tokens: 1D Tensor of Token outputs.
        Returns:
            A list of words.
        """
        words = []
        for token in tokens:
            word = self.idx2word[str(token.item())]
            if word == self.SYM_EOS:
                break
            if word not in [self.SYM_PAD, self.SYM_SOQ,
                            self.SYM_SOR, self.SYM_EOS]:
                words.append(word)
        sentence = str(' '.join(words))
        return sentence


def get_FastText_embedding(name, embed_size, vocab, questions=None):
    """Construct embedding tensor.
    Args:
        name (str): Which FastText embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            FastText word embeddings.
    """

    ft = torchtext.vocab.FastText('en', max_vectors=str(embed_size))
    #list_vocab = [[v] for v in vocab.idx2word.values()]  # TODO: use full questions
    """list_questions = []
    f = open(questions)
    question = json.load(f)
    for q in question["data"]:
        list_questions.append(q["question"].split(" "))
    f.close()"""
    #ft = FastText(list_questions, size=embed_size, min_count=1, window=5, sg=1, sample=1, iter=10)

    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)

    for i in range(vocab_size):
        try:
            embedding[i] = torch.tensor(ft[vocab.idx2word[str(i)]])
        except:
            pass

    return embedding


# ===========================================================
# Helpers.
# ===========================================================

def process_lengths(inputs, pad=0):
    """Calculates the lenght of all the sequences in inputs.
    Args:
        inputs: A batch of tensors containing the question or response
            sequences.
    Returns: A list of their lengths.
    """
    max_length = inputs.size(1)
    if inputs.size(0) == 1:
        lengths = list(max_length - inputs.data.eq(pad).sum(1))
    else:
        lengths = list(max_length - inputs.data.eq(pad).sum(1).squeeze())
    return lengths





