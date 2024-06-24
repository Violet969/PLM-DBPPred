# PLM-DBPPred
Cite the code:[![DOI](https://zenodo.org/badge/654626858.svg)](https://zenodo.org/doi/10.5281/zenodo.10675351)

What is PLM-DBPPred?
=============
A DNA binding protein prediction tool based on proteinBERT, ESM2_30 and protT5.

Env
=============
ProteinBERT environment https://github.com/nadavbra/protein_bert.

ESM2 environment https://github.com/facebookresearch/esm.

protT5 environment https://github.com/HannesStark/protein-localization.
```
# python==3.8.0 Pytorch==2.0.1 CUDA Version: 11.4 
conda env create -f env.yml
conda activate PLM_DBPPred
pip install -r requirements.txt
pip install bio-embeddings
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git
```
Download model Parameters
=============
Training model paramaters


Using PLM-DBPPred
=============
ProteinBERT

Training
```
python train.py -train_set ../DBP_dataset/DBP_Predict_refine_13289_230726.train.csv -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -exp_name test_py
```
Predict
```
python predict.py -test_set ./DBP_dataset/DBP_Predict_PDB.test.csv -o ./ -m ./DBP_model_param/
```

ProtT5

Used the bio-embedding to generate the .h5 file
```
bio_embeddings light_attention.yml
```
Training
```
python train.py --config ./configs/DBP_light_attention_train.yaml
```
Predict
```
python predict.py --config ./configs/DBP-test-predict.yaml  
```

ESM

Training
```
python train.py -train_set ../DBP_dataset/DBP_Predict_refine_13289_230726.train.csv -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -exp_name outputs
```
Predict
```
python predict.py -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv  -o run_result/ -m ESM2_30_model_param.t7
```
