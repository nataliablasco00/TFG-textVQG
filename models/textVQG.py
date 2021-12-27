"""Contains code for the IQ model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP



class textVQG(nn.Module):
    """Information Maximization question generation.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                  sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0,
                 encoder_max_len=None, num_att_layers=2, att_ff_size=300,
                 embedding=None, z_size=20, no_answer_recon=False,
                 no_image_recon=False, no_category_space=False):
        """Constructor for textVQG.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            encoder_max_len: Maximum length of encoder.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
            embedding (vocab_size, hidden_size): Tensor of embeddings or
                None. If None, embeddings are learned.
            z_size: Dimensions of noise epsilon.
        """
        super(textVQG, self).__init__()
        self.answer_recon = not no_answer_recon
        self.hidden_size = hidden_size
        if encoder_max_len is None:
            encoder_max_len = max_len
        self.num_layers = num_layers

        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(hidden_size)


        # Setup answer encoder.
        self.answer_encoder = EncoderRNN(vocab_size, max_len, hidden_size,
                                         input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p,
                                         n_layers=num_layers,
                                         bidirectional=False,
                                         rnn_cell=rnn_cell,
                                         variable_lengths=True)

        # Setup stacked attention to combine image and answer features.
        self.answer_attention = MLP(3*hidden_size, att_ff_size, hidden_size,
                                    num_layers=num_att_layers)

        # Setup question decoder.
        self.z_decoder = nn.Linear(z_size, hidden_size)
        self.gen_decoder = MLP(hidden_size, att_ff_size, hidden_size,
                               num_layers=num_att_layers)
        """self.decoder = DecoderRNN(vocab_size, max_len, hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  embedding=embedding)"""
        self.decoder = Transformer(
            num_tokens=vocab_size,
            dim_model=128,
            d_hid=512,
            num_heads=4, # 16
            num_layers=3, # 24,
            dropout_p=0.1,
            n_positions=max_len
        )



   
        

        # Setup answer reconstruction.
        if self.answer_recon:
            self.answer_reconstructor = MLP(
                    z_size, att_ff_size, hidden_size,
                    num_layers=num_att_layers)

    def flatten_parameters(self):
        if hasattr(self, 'decoder'):
            self.decoder.rnn.flatten_parameters()
        if hasattr(self, 'encoder'):
            self.encoder.rnn.flatten_parameters()

    def generator_parameters(self):
        params = self.parameters()
        params = filter(lambda p: p.requires_grad, params)
        return params

    def info_parameters(self):
        params = (list(self.answer_attention.parameters()))

        # Reconstruction parameters.
        
        if self.answer_recon:
            params += list(self.answer_reconstructor.parameters())

        params = filter(lambda p: p.requires_grad, params)
        return params


    def modify_hidden(self, func, hidden, rnn_cell):
        """Applies the function func to the hidden representation.

        This method is useful because some RNNs like LSTMs have a tuples.

        Args:
            func: A function to apply to the hidden representation.
            hidden: A RNN (or LSTM or GRU) representation.
            rnn_cell: One of RNN, LSTM or GRU.

        Returns:
            func(hidden).
        """
        if rnn_cell is nn.LSTM:
            return (func(hidden[0]), func(hidden[1]))
        return func(hidden)

    def parse_outputs_to_tokens(self, outputs):
        """Converts model outputs to tokens.

        Args:
            outputs: Model outputs.

        Returns:
            A tensor of batch_size X max_len.
        """
        # Take argmax for each timestep
        # Output is list of MAX_LEN containing BATCH_SIZE * VOCAB_SIZE.

        # BATCH_SIZE * VOCAB_SIZE -> BATCH_SIZE
        outputs = [o.max(1)[1] for o in outputs]

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs

    def encode_images(self, images):
        """Encodes images.

        Args:
            images: Batch of image Tensors.

        Returns:
            Batch of image features.
        """
        
        return self.encoder_cnn(images)

 

    def encode_answers(self, answers, alengths):
        """Encodes the answers.

        Args:

        Returns:
            batch of answer features.
        """
        # print(answers,alengths)
       
        _, encoder_hidden = self.answer_encoder(
                answers, alengths, None)
        if self.answer_encoder.rnn_cell == nn.LSTM:
            encoder_hidden = encoder_hidden[0]

        # Pick the hidden vector from the top layer.
        encoder_hidden = encoder_hidden[-1, :, :].squeeze()
        return encoder_hidden

    def encode_position(self, bbox):
        """Encodes the ocr positions.
        Returns batch of ocr token positions
        """
        # print(len(bbox))

        return bbox

    def encode_into_z(self, image_features, answer_features, bbox):
        """Encodes the attended features into z space.

        Args:
            image_features: Batch of image features.
            answer_features: Batch of answer features.

        Returns:
            mus and logvars of the batch.
        """
        # print("image_features:--", answer_features.shape)
        bbox1 = bbox.repeat(1, 128)
        together = torch.cat((image_features, answer_features, bbox1.float()), dim=1)
        #together = torch.cat((image_features, answer_features), dim=1)
        # print(together.shape)
        attended_hiddens = self.answer_attention(together)
        return attended_hiddens


    def decode_questions(self, image_features, zs,
                         questions=None, teacher_forcing_ratio=0,
                         decode_function=F.log_softmax):
        """Decodes the question from the latent space.

        Args:
            image_features: Batch of image features.
            zs: Batch of latent space representations.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.
        """
        batch_size = zs.size(0)
        z_hiddens = self.z_decoder(zs)
        if image_features is None:
            hiddens = z_hiddens
        else:
            hiddens = self.gen_decoder(image_features + z_hiddens)

        # Reshape encoder_hidden (NUM_LAYERS * N * HIDDEN_SIZE).
        hiddens = hiddens.view((1, batch_size, self.hidden_size))
        hiddens = hiddens.expand((self.num_layers, batch_size,
                                  self.hidden_size)).contiguous()
        if self.decoder.rnn_cell is nn.LSTM:
            hiddens = (hiddens, hiddens)
        result = self.decoder(inputs=questions,
                              encoder_hidden=hiddens,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def forward(self, images, answers, categories, alengths=None, questions=None,
                teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        """Passes the image and the question through a model and generates answers.

        Args:
            images: Batch of image Variables.
            answers: Batch of answer Variables.
            categories: Batch of answer Variables.
            alengths: List of answer lengths.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # features is (N * HIDDEN_SIZE)
        image_features = self.encode_images(images)

        # encoder_hidden is (N * HIDDEN_SIZE).
        answer_hiddens = self.encode_answers(answers, alengths)

        # Calculate the mus and logvars.
        zs = self.encode_into_z(image_features, answer_hiddens, bbox)
      
        result = self.decode_questions(image_features, zs,
                                       questions=questions,
                                       decode_function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return result

    def reconstruct_inputs(self, image_features, answer_features,bbox):
        """

        Args:
            answer_features: Batch of answer features.

        Returns:
            Reconstructed answer features.
        """
       
        recon_answer_features = None
        zs = self.encode_into_z(image_features, answer_features, bbox)
        if self.answer_recon:
            recon_answer_features = self.answer_reconstructor(zs)
        return recon_answer_features

    def encode_from_answer(self, images, answers,bbox, lengths=None):
        """

        Args:
            images: Batch of image Tensor.
            answers: Batch of answer Tensors.
            alengths: List of answer lengths.

        Returns:
            Batch of latent space encodings.
        """
        image_features = self.encode_images(images)
        answer_hiddens = self.encode_answers(answers, lengths)
        zs = self.encode_into_z(image_features, answer_hiddens, bbox)
        
        return image_features, zs



    def predict_from_answer(self, images, answers,bbox, lengths=None,
                            questions=None, teacher_forcing_ratio=0,
                            decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            images: Batch of image Tensors.
            answers: Batch of answer Tensors.
            alengths: List of answer lengths.
            questions: Batch of question Tensors.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        image_features, zs = self.encode_from_answer(images, answers,bbox, lengths=lengths)
        outputs, _, _ = self.decode_questions(image_features, zs, questions=questions,
                                              decode_function=decode_function,
                                              teacher_forcing_ratio=teacher_forcing_ratio)
        return self.parse_outputs_to_tokens(outputs)
