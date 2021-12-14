# textVQG

**Implementation of the paper Look, Read and Ask: Learning to Ask Questions by Reading Text in Images (ICDAR-2021)**
(Resolving some issues with respect to evaluation code. [Issue with NLG_eval and evaluate_textvqg.py code also, the model and entire code structure was previously written in python 2.7 environment and now it is being changed to python 3.8. Will update the complete code sooner.])

[paper](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_22)

## Requirements

* Use **pytorch 1.7.0 CUDA 10.2**

* Other requirements from 'requirements.txt'

**To setup environment**
```
# create new env 
$ virtualenv -p python2.7 textvqg

# activate 
$ source textvqg/bin/activate

# install pytorch, torchvision
$ conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

# install other dependencies
$ pip install -r requirements.txt
```

## Model Training
```
# Create the vocabulary files required for textVQG.
python utils/vocab.py

# Create the hdf5 dataset.
python utils/store_dataset.py

# Train the model.
python train_textvqg.py

# Evaluate the model.
python evaluate_textvqg.py
```

## Results
The following are some results of the proposed method:

![res1](https://user-images.githubusercontent.com/44959352/132222886-1fd59167-772d-457c-b0b9-165cc81ea25d.png)
![res2](https://user-images.githubusercontent.com/44959352/132222906-1ee94d1f-ce27-483a-8368-ad0b6b1b38a1.png)



## Cite
If you find this code/paper  useful for your research, please consider citing.
```
@InProceedings{10.1007/978-3-030-86549-8_22,
author="Jahagirdar, Soumya
and Gangisetty, Shankar
and Mishra, Anand",
editor="Llad{\'o}s, Josep
and Lopresti, Daniel
and Uchida, Seiichi",
title="Look, Read and Ask: Learning to Ask Questions by Reading Text in Images",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="335--349"
}


```


## Acknowledgements
This repo uses few utility function provided by https://github.com/ranjaykrishna/iq.

### Contact
For any clarification, comment, or suggestion please create an issue or contact [Soumya Shamarao Jahagirdar](https://www.linkedin.com/in/soumya-jahagirdar/).



