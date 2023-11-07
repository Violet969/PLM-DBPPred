# PLM-DBPPred

What is PLM-DBPPred?
=============
A DNA binding protein prediction tool based on proteinBERT, ESM2_30 and protT5.

Env
=============
Install ProteinBERT environment https://github.com/nadavbra/protein_bert.
* tensorflow (2.4.0)
* tensorflow_addons (0.12.1)
* numpy (1.20.1)
* pandas (1.2.3)
* h5py (3.2.1)
* lxml (4.3.2)
* pyfaidx (0.5.8)

Download model Parameters
=============
Training model paramaters
https://huggingface.co/Violet969/PLM-DBPPred/tree/main/PLM-DBPPred/DBP_model_param

Using PLM-DBPPred
=============
Training
1. Open train_PLM-DBPPred.ipynb.
2. Change the train and test sets file path.
   
Test
1. Open test_PLM-DBPPred.ipynb.
2. Change the test sets file path.

Prediction
1. Use the data.ipynb to transfer fasta to csv.
2. Open predict_PLM-DBPPred.ipynb.
3. Change the test_set file path.



   
