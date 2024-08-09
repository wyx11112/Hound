# Hound: Hunting Supervision Signals for Few and Zero Shot Node Classification on Text-attributed Graph


- dataset/: the directory of data sets. Currently, it only has the dataset of Cora, if you want the other *four processed Amazon datasets*, you can download and put them under this directory, the link is https://drive.google.com/drive/folders/1b69yIbWBiSpQVS3oTonFqVr1dB-CV51t?usp=sharing


# For pre-training:
Few-shot pre-training on Cora dataset,

    python main_train.py --pretrain_model few --loss con_sum_np

Zero-shot pre-training on Cora dataset,

    python main_train.py --pretrain_model zero --loss con_sum_mar_np


If on Amazon datasets, few-shot pre-training should be:

    python main_train_amazon.py --data_name dataName --pretrain_model few --loss con_sum_np

Similarly, for zero-shot pre-training:

    python main_train_amazon.py --data_name dataName --pretrain_model zero --loss con_sum_mar_np

# For few-shot testing:
On Cora dataset,

    python main_test.py

If on Amazon datasets, it should be:

    python main_test_amazon.py


# For zero-shot testing:
On Cora dataset,

    python zero-shot-cora.py 

If on Amazon datasets, it should be:

    python zero-shot-amazon.py
