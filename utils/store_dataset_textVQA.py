"""Transform all the Scene Text VQA dataset if nto a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vocab import load_vocab, process_text

def create_answer_mapping(annotations, questions):

    # data = annotations
    answers = {}
    image_ids = set()
    for q in questions["data"]:
        question_id = q.get("image_id")+"-"+" ".join(q.get("question"))
        answer = q.get("answers")[0]
        answers[question_id] = answer
        image_ids.add(q.get("image_id"))
    return answers, image_ids


def save_dataset(image_dir, questions, OCR, vocab,output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False):

    # Load the data.
    max_ocr_len=4
    vocab = load_vocab(vocab)
    with open(OCR) as f:
        ocr = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    # Get the mappings from qid to answers.
    print(" -->", len(ocr["data"]), len(questions["data"]))
    qid2ans, image_ids = create_answer_mapping(ocr, questions)
    total_questions = len(questions["data"])+1
    total_images = len(image_ids)
    print ("Number of images to be written: %d" % total_images)
    print ("Number of QAs to be written: %d" % total_questions)

    total_questions = 698#5043
    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions + 1, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions + 1,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images + 1, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions + 1, max_a_length), dtype='i')
    d_ocr_positions = h5file.create_dataset(
        "ocr_positions", (total_questions + 1, max_ocr_len), dtype='f')


    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    # bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 0
    q_index = 0
    done_img2idx = {}
    img_shapes = {}
    img_paths = {}
    for idx, entry in enumerate(questions["data"]):
        print(q_index)
        image_id = entry.get("image_id")
        question_id = entry.get("image_id")+"-"+" ".join(entry.get("question"))

        bbox = []
        for r in ocr["data"]:
            if image_id == r["image_id"]:
                for s1 in entry["answers"][0].split(" "):
                    for s2 in r["ocr_info"]:
                        if s1 == s2["word"]:
                            aux = list(s2["bounding_box"].values())
                            bbox.append([aux[-2], aux[-1], aux[0], aux[1]])
                break

        if len(bbox) == 0:
            continue

        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                path = "./" + (image_id)
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            except IOError:
                path = "./" + (image_id)
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            img_shapes[image_id] = np.array(image).shape
            image = transform(image)


            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            img_paths[i_index] = path
            i_index += 1

        #process_ocr_pos = [x["bbox"] for x in entry["ans_bboxes"]]
        v0, v1, v2, v3 = float('inf'), float('inf'), 0, 0
        for x in bbox:
            v0 = min(v0, x[0])
            v1 = min(v1, x[1])
            v2 = max(v2, x[0]+x[2])
            v3 = max(v3, x[1]+x[3])

        # print(entry.get("ocr_position"))
        s = img_shapes[image_id]
        d_ocr_positions[q_index, :4] = [float(v0), float(v1), float(v2), float(v3)]
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
        d_questions[q_index, :length] = q
        # print(q, "----" ,d_questions[q_index, :length],"len is: ", length)
        answer = qid2ans[question_id]
        ans, length = process_text(answer, vocab,
                                 max_length=max_a_length)

        d_answers[q_index, :length] = ans

        d_indices[q_index] = done_img2idx[image_id]
        q_index += 1
        #if idx >= total_questions:
        #    break

    # bar.update(q_index)
    h5file.close()
    print ("Number of images written: %d" % i_index)
    print ("Number of QAs written: %d" % q_index)
    with open('img_paths.json', 'w') as fp:
        json.dump(img_paths, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/data/train_images',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqa_qa_ocr_data.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqa_qa_ocr_data.json',
                        help='Path for train annotation file.')

    parser.add_argument('--vocab-path', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/vocab_iq1.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqg_dataset1.hdf5',
                        help='directory for resized images.')


    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')
    args = parser.parse_args()


    save_dataset(args.image_dir, args.questions, args.annotations, args.vocab_path,
                 args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    Vocabulary()