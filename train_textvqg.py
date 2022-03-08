"""This script is used to generate text based visual question generator
"""



import argparse
import json
import logging
import os
import random
import time
import torch.nn as nn
import wandb
# from utils import NLGEval
from models import textVQG
from utils import Vocabulary
from utils import get_FastText_embedding
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

wandb.init(project="textVQG-noBB-Analysis", entity="natblan")



def create_model(args, vocab, embedding=None):
    """Creates the model.

    Args:
        args: Instance of Argument Parser.
        vocab: Instance of Vocabulary.

    Returns:
        An textVQG model.
    """
    # Load FastText embedding.
    if args.use_FastText:
        embedding = get_FastText_embedding(args.embedding_name,
                                        args.hidden_size,
                                        vocab, args.questions)
    else:
        embedding = None

    # Build the models
    logging.info('Creating textVQG model...')
    textvqg = textVQG(len(vocab), args.max_length, args.hidden_size,
             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=args.num_layers,
             rnn_cell=args.rnn_cell,
             dropout_p=args.dropout_p,
             input_dropout_p=args.input_dropout_p,
             encoder_max_len=args.encoder_max_len,
             embedding=embedding,
             num_att_layers=args.num_att_layers,
             z_size=args.z_size,
             no_answer_recon=args.no_answer_recon)
    return textvqg


def evaluate(textvqg, data_loader, criterion, l2_criterion, args):
    """Calculates text vqg average loss on data_loader.

    Args:
        text vqg: text-based visual question generation model.
        data_loader: Iterator for the data.
        criterion: The loss function used to evaluate the loss.
        l2_criterion: The loss function used to evaluate the l2 loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    textvqg.eval()
    total_gen_loss = 0.0
    total_info_loss = 0.0
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for iterations, (images, questions, answers, qindices, bbox, _) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            qindices = qindices.cuda()
            bbox = bbox.cuda()
        
        images = images[:-1]
        questions = questions[:-1]
        answers = answers[:-1]
        qindices = qindices[:-1]
        bbox = bbox[:-1]
        alengths = process_lengths(answers)
        # Forward, Backward and Optimize
        image_features = textvqg.encode_images(images)
        answer_features = textvqg.encode_answers(answers, alengths)
        position_features = textvqg.encode_position(bbox)
        zs = textvqg.encode_into_z(image_features, answer_features, bbox)

        outputs = textvqg.decode_questions(
                image_features, zs, questions=questions,
                teacher_forcing_ratio=1.0)

        # Reorder the questions based on length.
        # questions = torch.index_select(questions, 0, qindices)

        # Ignoring the start token.
        questions = questions[:, 1:]
        qlengths = process_lengths(questions)

        # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
        # (BATCH x MAX_LEN x VOCAB).
        outputs = [o.unsqueeze(1) for o in outputs]
        outputs = torch.cat(outputs, dim=1)
        # outputs = torch.index_select(outputs, 0, qindices)

        # Calculate the loss.
        targets = pack_padded_sequence(questions, qlengths,
                                       batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, qlengths,
                                       batch_first=True)[0]
        gen_loss = criterion(outputs, targets)
        total_gen_loss += gen_loss.data.item()


        # Reconstruction.
        if not args.no_image_recon or not args.no_answer_recon:
            image_targets = image_features.detach()
            answer_targets = answer_features.detach()
            recon_answer_features = textvqg.reconstruct_inputs(
                    image_targets, answer_targets,bbox)
            
            if not args.no_answer_recon:
                recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
                

        # Quit after eval_steps.
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

        # Print logs
        if iterations % args.log_step == 0:
             delta_time = time.time() - start_time
             start_time = time.time()
             # logging.info('Time: %.4f, Step [%d/%d], gen loss: %.4f, '
             #              '  A-recon: %.4f, '
        
             #             % (delta_time, iterations, 
             #                total_gen_loss/(iterations+1),
             #                ))
    
    return total_gen_loss / (iterations+1), total_info_loss / (iterations + 1)


def run_eval(textvqg, data_loader, criterion, l2_criterion, args, epoch,
             scheduler, info_scheduler):
    logging.info('=' * 80)
    start_time = time.time()
    val_gen_loss, val_info_loss = evaluate(
            textvqg, data_loader, criterion, l2_criterion, args)
    delta_time = time.time() - start_time
    scheduler.step(val_gen_loss)
    scheduler.step(val_info_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val-gen-loss: %.4f, '
                 'Val-info-loss: %.4f' % (
        delta_time, epoch, args.num_epochs, val_gen_loss, val_info_loss))
    logging.info('=' * 80)



def compare_outputs(images, questions, answers, bbox,
                    alengths, textvqg, vocab, logging,
                    args, num_show=8):
    """Sanity check generated output as we train.

    Args:
        images: Tensor containing images.
        questions: Tensor containing questions as indices.
        answers: Tensor containing answers as indices.
        alengths: list of answer lengths.
        vqg: A question generation instance.
        vocab: An instance of Vocabulary.
        logging: logging to use to report results.
    """
    textvqg.eval()
    # Forward pass through the model.
    outputs = textvqg.predict_from_answer(images, answers,bbox, questions=questions, lengths=alengths)
    l = []
    for _ in range(num_show):
        logging.info("         ")
        i = random.randint(0, images.size(0) - 1)  # Inclusive

        # Log the outputs.
        output = vocab.tokens_to_words(outputs[i])
        question = vocab.tokens_to_words(questions[i])
        answer = vocab.tokens_to_words(answers[i])
        logging.info('Generated question : %s\n'
                     'Target  question (%s): -> %s'
                     % (output, 
                        question, answer))
        logging.info("         ")

    for i in range(len(questions)):
        output = vocab.tokens_to_words(outputs[i])
        l.append(output)
    return l



def train(args):

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_path, 'train1.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)


    # Build data loader
    logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
   
   
    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset, transform,
                                     args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     max_examples=args.max_examples,
                                     sampler=val_sampler)
    logging.info("Done")

    textvqg = create_model(args, vocab)
    if args.load_model is not None:
        textvqg.load_state_dict(torch.load(args.load_model))
    logging.info("Done")

    # Loss criterion.
    pad = vocab(vocab.SYM_PAD)  # Set loss weight for 'pad' symbol to 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad) # NLLoss
    l2_criterion = nn.MSELoss()

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        textvqg.cuda()
        criterion.cuda()
        l2_criterion.cuda()

    # Parameters to train.
    gen_params = textvqg.generator_parameters()
    info_params = textvqg.info_parameters()
    learning_rate = args.learning_rate
    info_learning_rate = args.info_learning_rate
    gen_optimizer = torch.optim.Adam(gen_params, lr=learning_rate)
    info_optimizer = torch.optim.Adam(info_params, lr=info_learning_rate)
    factor = 0.99 # TODO: Afegir per terminal
    min_lr = 1e-7
    scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=factor, patience=args.patience,
                                  verbose=True, min_lr=min_lr)
    info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
                                       factor=factor, patience=args.patience,
                                       verbose=True, min_lr=min_lr)


    wandb.config = {"learning_rate": learning_rate,
                    "epochs": args.num_epochs,
                    "factor_ReduceLROnPlateau": factor,
                    "min_lr": min_lr}

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0

    # Optional losses. Initialized here for logging.
    recon_answer_loss = 0.0
  
    total_info_loss = 0.0
  
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers, qindices, bbox, _) in enumerate(data_loader):
            n_steps += 1
            images=images[:-1]
            questions=questions[:-1]
            answers=((answers[:-1]))
            # print(answers)

            qindices = qindices[:-1]
            # print(qindices)
            bbox = bbox[:-1]
            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                qindices = qindices.cuda()
                bbox = bbox.cuda()
            alengths = process_lengths(answers)

            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                run_eval(textvqg, val_data_loader, criterion, l2_criterion,
                         args, epoch, scheduler, info_scheduler)
            generated_questions = compare_outputs(images, questions, answers, bbox,
                            alengths, textvqg, vocab, logging,  args)

            # Forward.
            textvqg.train()
            gen_optimizer.zero_grad()
            info_optimizer.zero_grad()
            image_features = textvqg.encode_images(images)
            # print(answers, alengths)bb
            answer_features = textvqg.encode_answers(answers, alengths)
            ocr_token_pos = textvqg.encode_position(bbox)
            #print("answer features: ",answer_features.size())
            # Question generation.
            zs = textvqg.encode_into_z(image_features, answer_features, bbox)
           
            outputs = textvqg.decode_questions(
                    image_features, zs, questions=questions,
                    teacher_forcing_ratio=1.0)
            # Reorder the questions based on length.
            # questions = torch.index_select(questions, 0, qindices)
           
            # Ignoring the start token.
            questions = questions[:, 1:]
            qlengths = process_lengths(questions)

            # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
            # (BATCH x MAX_LEN x VOCAB).
            outputs = [o.unsqueeze(1) for o in outputs]
           
            outputs = torch.cat(outputs, dim=1)
           
            # outputs = torch.index_select(outputs, 0, qindices)
            # print("outputs: ", outputs.size())
            # Calculate the generation loss.
            targets = pack_padded_sequence(questions, qlengths,
                                           batch_first=True)[0]
                                   
            outputs = pack_padded_sequence(outputs, qlengths,
                                           batch_first=True)[0]
            # print("target size: ",targets.size(),"----","output size: ",outputs.size())
            #break;
            gen_loss = criterion(outputs, targets)
            total_loss = 0.0
            total_loss += args.lambda_gen * gen_loss
            gen_loss = gen_loss.item()

            wandb.log({"loss": total_loss})
            wandb.log({"number": len(set(generated_questions))})

            # Generator Backprop.
            total_loss.backward()
            gen_optimizer.step()

            # Reconstruction loss.
            recon_answer_loss = 0.0
            if not args.no_answer_recon:
                total_info_loss = 0.0
                gen_optimizer.zero_grad()
                info_optimizer.zero_grad()
                image_targets = image_features.detach()
                answer_targets = answer_features.detach()
                
                recon_answer_features = textvqg.reconstruct_inputs(
                         image_targets,answer_targets,bbox)

                # Answer reconstruction loss.
                if not args.no_answer_recon:
                    recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
                    total_info_loss += args.lambda_a * recon_a_loss
                    recon_answer_loss = recon_a_loss.item()

                # Info backprop.
                total_info_loss.backward()
                info_optimizer.step()


            # Print log info
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'LR: %f, gen: %.4f,  '
                             ' A-recon: %.4f, '
                             
                             % (delta_time, epoch, args.num_epochs, i,
                                total_steps, gen_optimizer.param_groups[0]['lr'],
                                gen_loss,  recon_answer_loss
                                ))

            
            # Save the models
            if args.save_step is not None and (i+1) % args.save_step == 0:
                torch.save(textvqg.state_dict(),
                           os.path.join(args.model_path,
                                        'textvqg-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))
            compare_outputs(images, questions, answers, bbox,
                            alengths, textvqg, vocab, logging,  args)

        torch.save(textvqg.state_dict(),
                   os.path.join(args.model_path,
                                'textvqg-tf-%d.pkl' % (epoch+1)))
        # Evaluation and learning rate updates.
        run_eval(textvqg, val_data_loader, criterion, l2_criterion,
                 args, epoch, scheduler, info_scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='random/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=300,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--info-learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Lambda values.
    parser.add_argument('--lambda-gen', type=float, default=1.0,
                        help='coefficient to be added in front of the generation loss.')
    parser.add_argument('--lambda-z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    parser.add_argument('--lambda-t', type=float, default=0.0001,
                        help='coefficient to be added with the type space loss.')
    parser.add_argument('--lambda-a', type=float, default=0.01,
                        help='coefficient to be added with the answer recon loss.')
    parser.add_argument('--lambda-i', type=float, default=0.0001,
                        help='coefficient to be added with the image recon loss.')
    parser.add_argument('--lambda-z-t', type=float, default=0.001,
                        help='coefficient to be added with the t and z space loss.')

    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/vocab_iq1.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqg_dataset1.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqg_dataset1.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')
    parser.add_argument('--questions', type=str, default=None,
                        help='Path to file with questions')


    # Model parameters
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers in lstm.')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder-max-len', type=int, default=4,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use-FastText', action='store_true',
                        help='Whether to use FastText embeddings.')
    parser.add_argument('--embedding-name', type=str, default='6B',
                        help='Name of the FastText embedding to use.')
    parser.add_argument('--dropout-p', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input-dropout-p', type=float, default=0.3,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num-att-layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z-size', type=int, default=512,
                        help='Dimensions to use for hidden variational space.')

    # Ablations.
    parser.add_argument('--no-image-recon', action='store_true', default=False,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no-answer-recon', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-category-space', action='store_true', default=True,
                        help='Does not try to reconstruct answer.')

    args = parser.parse_args()
    train(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
