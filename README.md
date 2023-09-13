# Diffusion-GNNs
Graph Neural Networks for Link Prediction with Learnable Diffusion Distances
## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

or

```
conda env create -f conda_graphy_environment.yml
conda activate graphy
```
## Code organization
* `main.py`: script with inline arguments for running the experiments.
* `models.py`: script with our proposed architecture.
* `pump.py`: implementation of the proposed **pump**.
* `utils.py`: extra functions used for the experiments.
* `sota.py`: script with the implementation of the baselines (VGAE and ARGA).
## Run experiments
```python
python main.py --dataset wisconsin --hidden_channels 1024 --num_centers 30  --epochs 200 --lr 0.00001
python main.py --dataset cornell --hidden_channels 1024 --num_centers 30  --epochs 100 --lr 0.0001  --dropout 0.15
python main.py --dataset chamaleon --hidden_channels 1024 --num_centers 5  --epochs 1000 --lr 0.0005  --dropout 0.15
python main.py --dataset squirrel --hidden_channels 1024 --num_centers 10  --epochs 1000 --lr 0.0001 --dropout 0
python main.py --dataset cora --hidden_channels 1024 --num_centers 3  --epochs 500 --lr 0.00001
```
## Hyperparameters Settings
| Dataset     | Hidden Channels | Dropout | Learning Rate | Weight Decay | k/\#Jumps |
|-------------|-----------------|---------|---------------|--------------|-----------|
| Texas       | 64              | 0.2     | 0.03          | 0.0005       | 20        |
| Wisconsin   | 64              | 0.5     | 0.03          | 0.0005       | 5         |
| Cornell     | 128             | 0.5     | 0.03          | 0.001        | 5         |
| Actor       | 16              | 0.2     | 0.03          | 0.0001       | 3         |
| Squirrel    | 128             | 0.5     | 0.003         | 0.0005       | 8         |
| Chameleon   | 128             | 0.35    | 0.003         | 0.0005       | 12        |
| Citeseer    | 128             | 0.5     | 0.003         | 0.0005       | 5         |
| Pubmed      | 128             | 0.3     | 0.01          | 0.0005       | 3         |
| Cora        | 128             | 0.5     | 0.002         | 0.0005       | 5         |
| Penn94      | 16              | 0.5     | 0.001         | 0.0001       | 3         |
| Ogbn-arXiv  | 128             | 0.3     | 0.01          | 0.0005       | 3         |
| ArXiv-year  | 128             | 0.2     | 0.003         | 0.0005       | 3         |

