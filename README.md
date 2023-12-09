# PLM-DBPPred

What is PLM-DBPPred?
=============
A DNA binding protein prediction tool based on proteinBERT, ESM2_30 and protT5.

Env
=============
ProteinBERT environment https://github.com/nadavbra/protein_bert.

ESM2 environment https://github.com/facebookresearch/esm.

protT5 environment https://github.com/HannesStark/protein-localization.
```
conda env create -f env.yml
pip install -r requirements.txt
conda activate PLM_DBPPred
```
Download model Parameters
=============
Training model paramaters


Using PLM-DBPPred
=============
ProteinBERT
Training
1. Open train_PLM-DBPPred.ipynb.
2. Change the train and test sets file path.

Prediction
1. Use the data.ipynb to transfer fasta to csv.
2. Open predict_PLM-DBPPred.ipynb.
3. Change the test_set file path.

ProtT5
Used the bio-embedding to generate the .h5 file
```
bio_embeddings light_attention.yaml
```
Training
```
python train.py --config ./configs/DBP_light_attention_train.yaml
```
Test
```
python predict.py --config ./configs/DBP-test-predict.yaml  
```

ESM
Training
```
python train.py -train_set ../DBP_dataset/DBP_Predict_refine_13289_230726.train.csv -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -exp_name outputs
```
Test
```
python predict.py -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv  -o run_result/ -m ESM2_30_model_param.t7
```
