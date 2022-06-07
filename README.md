# textVQG

This repository has used as baseline code from: ([https://link.springer.com/chapter/10.1007/978-3-030-86549-8_22](https://github.com/soumyasj/textVQG))


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
