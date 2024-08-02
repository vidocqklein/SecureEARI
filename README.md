# EARI

## Dependency
- Python 3.9
- PyTorch 1.11.0
- torch-geometric 2.0.4
- CUDA 11.3
  
## Optional Datasets
- cora 
- citeseer 
- pubmed
- dblpv7 
- citationv1 
- acmv9

## Running the Code
We do not provide the trained model since the training cost for each experiment is acceptable. You can train the model by yourself by running the following command.
For example, to train the EARi model on the Cora and Dblpv7 dataset, you can run the following command:
```
python main.py cora --model eari --epochs 2000 --lr 0.01 --ptb_rate 0.03 --threshhold 0.8 --inductive_setting False --perturb_nodes_radio 0.5

python main.py dblpv7 --model eari --epochs 2000 --lr 0.01 --ptb_rate 0.05 --threshhold 0.8 --inductive_setting True --tdataset citationv1 --perturb_nodes_radio 0.5

```
