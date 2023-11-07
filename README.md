# PLM-DBPPred

What is PLM-DBPPred?
=============
A DNA binding protein prediction tool based on proteinBERT, ESM2_30 and protT5.

Env
=============
ProteinBERT environment https://github.com/nadavbra/protein_bert
ESM2 environment https://github.com/facebookresearch/esm
protT5 environment https://github.com/HannesStark/protein-localization

conda env create -f env.yml
conda activate PLM_DBPPred

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



   
