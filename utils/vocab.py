

"""Creates a vocabulary using iq_dataset for the vqa dataset.
"""

from collections import Counter
# from train_utils import Vocabulary

import argparse
import json
import logging
import nltk
import numpy as np
import re

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
        with open(location, 'wb') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    def load(self, location):
        with open(location, 'rb') as f:
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
        
def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.
    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.
    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))
    return np.array(output[:length]), length


def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.
    Args:
        vocab_path: The location of the vocab pickle file.
    Returns:
        A Vocabulary object.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path)
    return vocab


def tokenize(sentence):
    """Tokenizes a sentence into words.
    Args:
        sentence: A string of words.
    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    sentence = sentence.decode('utf8')
    #sentence = sentence.encode("ascii", "ignore").decode('utf8')
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    tokens1=[]
    for tok in tokens:
        tokens1.append(tok.encode('utf8'))
    return tokens1


def build_vocab(questions,  threshold):
    """Build a vocabulary from the annotations.
    Args:
        annotations: A json file containing the questions and answers.
        cat2ans: A json file containing answer types.
        threshold: The minimum number of times a work must occur. Otherwise it
            is treated as an `Vocabulary.SYM_UNK`.
    Returns:
        A Vocabulary object.
    """
    with open(questions) as f:
        questions = json.load(f)


    words = []


    counter = Counter()

    for entry in questions["data"]:
        qu = entry["question"]
        q_tokens = tokenize(qu.encode('utf8'))
        counter.update(q_tokens)
    for entry in questions["data"]:
        qu = entry["answers"][0]
        q_tokens = tokenize(qu.encode('utf8'))
        counter.update(q_tokens)
    print(counter)


    # If a word frequency is less than 'threshold', then the word is discarded.
    words.extend([word for word, cnt in counter.items() if cnt >= threshold])
    words = list(set(words))
    words.sort()
    vocab = create_vocab(words)
    return vocab


def create_vocab(words):
    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--questions', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqa_qa_ocr_data.json'
                       ,
                        help='Path for train questions file.')
    
    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=4,
                        help='Minimum word count threshold.')

    # Outputs.
    parser.add_argument('--vocab-path', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/vocab_iq.json',
                        help='Path for saving vocabulary wrapper.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(args.questions, args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)