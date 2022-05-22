"""This script is used to evaluate a model.
"""

from torchvision import transforms

import argparse
import json
import logging
import os
import progressbar
import torch

from models import textVQG
# from nlgeval import NLGEval
from utils import Dict2Obj
from utils import Vocabulary
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import get_FastText_embedding


def evaluate(textvqg, data_loader, vocab, args, params):
    """Runs BLEU, METEOR, CIDEr and distinct n-gram scores.

    Args:
        text vqg: text visual question generation model.
        data_loader: Iterator for the data.
        args: ArgumentParser object.
        params: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    textvqg.eval()
    # nlge = NLGEval(no_skipthoughts=True, no_FastText=True)
    preds = []
    gts = []
    idxs = []
    bar = progressbar.ProgressBar(maxval=len(data_loader)).start()
    # for iterations, (images, questions, answers, _) in enumerate(data_loader):  # TODO: commented
    for iterations, (images, questions, answers, qindices, bbox, img_indices) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            answers = answers.cuda()
            bbox = bbox.cuda()

        alengths = process_lengths(answers)

        # Predict.
        if args.from_answer:
            outputs = textvqg.predict_from_answer(images, answers, bbox, alengths)  # TODO: added bbox


        for i in range(images.size(0)):
            output = vocab.tokens_to_words(outputs[i])
            preds.append(output)

            question = vocab.tokens_to_words(questions[i])
            gts.append(question)
            idxs.append(int(img_indices[i]))
        bar.update(iterations)

    print('='*80)
    print('GROUND TRUTH')
    print(gts[:args.num_show])
    print('-'*80)
    print('PREDICTIONS')
    print(preds[:args.num_show])
    print('='*80)
    # scores = nlge.compute_metrics(ref_list=[gts], hyp_list=preds)
    return idxs, gts, preds


def main(args):
    """Loads the model and then calls evaluate().

    Args:
        args: Instance of ArgumentParser.
    """

    # Load the arguments.
    model_dir = os.path.dirname(args.model_path)
    params = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(model_dir, 'eval.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(params.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(params.vocab_path)

    # Build data loader
    logging.info("Building data loader...")

    # Load FastText embedding.
    if params.use_FastText:
        embedding = get_FastText_embedding(params.embedding_name,
                                        params.hidden_size,
                                        vocab)
    else:
        embedding = None

    # Build data loader
    logging.info("Building data loader...")

    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples)
    logging.info("Done")

    # Build the models
    logging.info('Creating textVQG model...')
    textvqg = textVQG(len(vocab), params.max_length, params.hidden_size,
             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=params.num_layers,
             rnn_cell=params.rnn_cell,
             dropout_p=params.dropout_p,
             input_dropout_p=params.input_dropout_p,
             encoder_max_len=params.encoder_max_len,
             embedding=embedding,
             num_att_layers=params.num_att_layers,
             z_size=params.z_size,
             )
    logging.info("Done")

    logging.info("Loading model.")
    textvqg.load_state_dict(torch.load(args.model_path))

    # Setup GPUs.

    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        textvqg.cuda()


    idx, gts, preds = evaluate(textvqg, data_loader, vocab, args, params)

    # Print and save the scores.
    with open(os.path.join(model_dir, args.indices_path), 'w') as indices_file:
        json.dump(idx, indices_file)
    with open(os.path.join(model_dir, args.preds_path), 'w') as preds_file:
        json.dump(preds, preds_file)
    with open(os.path.join(model_dir, args.gts_path), 'w') as gts_file:
        json.dump(gts, gts_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default = '/home/shankar/Desktop/VQA_REU/Stvqa/weights2/tf1/vqg-tf-8.pkl',
                        help='Path for loading trained models')
    parser.add_argument('--indices-path', type=str, default='/home/shankar/Desktop/VQA_REU/Stvqa/weights/results.json',
                        help='Path for saving results.')
    parser.add_argument('--preds-path', type=str, default='/home/shankar/Desktop/VQA_REU/Stvqa/weights/preds.json',
                        help='Path for saving predictions.')
    parser.add_argument('--gts-path', type=str, default='/home/shankar/Desktop/VQA_REU/Stvqa/weights/gts.json',
                        help='Path for saving ground truth.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='When set, only evalutes that many data points.')
    parser.add_argument('--num-show', type=int, default=10,
                        help='Number of predictions to print.')
    parser.add_argument('--from-answer', action='store_true', default=True,
                        help='When set, only evalutes textvqg model with answers;'
                        ' otherwise it tests textvqg with answer types.')

    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/textvqg_dataset.hdf5',
                        help='path for train annotation json file')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    Vocabulary()
