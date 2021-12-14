"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data


class textVQGDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, transform=True, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and token).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.ocr_positions = annos['ocr_positions']
            # print(self.ocr_positions)

        if self.indices is not None:
            index = self.indices[index]
        question = self.questions[index]
        answer = self.answers[index]
        # print(answer)
        ocr_pos = self.ocr_positions[index]
        # print("ocr_positions:----", ocr_pos)
        image_index = self.image_indices[index]
        image = self.images[image_index]

        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        # print("before", answer)
        ocr_pos = torch.from_numpy(ocr_pos)
        # print(ocr_pos)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        if self.transform is not None:
            image = self.transform(image)

        # print(answer, alength)
        return (ocr_pos,image, question, answer,
                qlength.item(), alength.item())

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        # print(annos['questions'].shape[0])
        return annos['questions'].shape[0]


def collate_fn(data):
    
    # Sort a data list by caption length (descending order).
    # print(type(data[0][0]))
    data.sort(key=lambda x: x[4], reverse=True)
    ocr_positions, images, questions, answers, qlengths ,_ = zip(*data)
    # print("after:",answers)
    # print(ocr_positions)
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    qindices = np.flip(np.argsort(qlengths), axis=0).copy()
    # print("qindices:-----", qindices)
    qindices = torch.Tensor(qindices).long()
    # print("qindices:-----", qindices)
    # print("positions:----",(ocr_positions))
    # ocr_positions = torch.Tensor(ocr_positions)
    # print("positions:----",(ocr_positions))
    ocr_positions = torch.stack(ocr_positions, 0).long()
    return images, questions, answers,  qindices, ocr_positions


def get_loader(dataset, transform, batch_size, sampler=None,
                   shuffle=True, num_workers=1, max_examples=None,
                   indices=None):
    
    textvqg = textVQGDataset(dataset, transform=transform, max_examples=max_examples,
                    indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=textvqg,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

