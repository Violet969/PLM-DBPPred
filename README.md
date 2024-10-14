# PLM-DBPPred
Cite the code:[![DOI](https://zenodo.org/badge/654626858.svg)](https://zenodo.org/doi/10.5281/zenodo.10675351)

What is PLM-DBPPred?
=============
A DNA binding protein prediction tool based on proteinBERT, ESM2_30 and protT5.
![image](https://github.com/Violet969/PLM-DBPPred/blob/main/PLM-DBPPred.jpg)

ProteinBERT Env
=============
ProteinBERT environment https://github.com/nadavbra/protein_bert.
```
# python==3.8.0 Tensorflow==2.5.0 CUDA Version: 11.4 
conda env create -f ProteinBERT_env.yml
conda activate PLM-DBPPred_ProteinBERT
```

Using ProteinBERT
=============
ProteinBERT

Training
```
python train.py -train_set ../DBP_dataset/DBP_Predict_refine_13289_230726.train.csv -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -exp_name test_py
```
Predict
```
cd proteinBERT
python predict.py -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -o ./ -m ./model_param/
```

ESM and ProtT5 Env
=============
ESM2 environment https://github.com/facebookresearch/esm.
protT5 environment https://github.com/HannesStark/protein-localization.
```
# python==3.8.0 Pytorch==2.0.1 CUDA Version: 11.4 
conda env create -f ESM_ProtT5_env.yml
conda activate PLM_DBPPred_ESM_ProtT5
pip install -r ESM_ProtT5_requirements.txt
pip install bio-embeddings[all]
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git
```

Using ProtT5
=============
ProtT5

Used the bio-embedding to generate the .h5 file
```
#Change the file path (sequences_file: ./Train_dataset.fasta prefix: ./Train_dataset_emb)
bio_embeddings light_attention.yml
```
Training
```
#Change the file path of .h5 and .fasta
python train.py --config ./configs/DBP_light_attention_train.yml
```
Predict
```
#Change the file path of .h5 and .fasta
python predict.py --config ./configs/DBP_test_predict.yml
```
Using ESM
=============
ESM

Training
```
python train.py -train_set ../DBP_dataset/DBP_Predict_refine_13289_230726.train.csv -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv -exp_name outputs
```
Predict
```
python predict.py -test_set ../DBP_dataset/DBP_Predict_PDB.test.csv  -o run_result/ -m ESM2_30_model_param.t7
```

## Citing PLM-DBPPred
```
@article{hu2024systematic,
  title={Systematic discovery of DNA-binding tandem repeat proteins<? mode longmeta?>},
  author={Hu, Xiaoxuan and Zhang, Xuechun and Sun, Wen and Liu, Chunhong and Deng, Pujuan and Cao, Yuanwei and Zhang, Chenze and Xu, Ning and Zhang, Tongtong and Zhang, Yong E and others},
  journal={Nucleic Acids Research},
  pages={gkae710},
  year={2024},
  publisher={Oxford University Press}
}
```

