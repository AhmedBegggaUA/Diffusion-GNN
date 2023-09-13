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
| Dataset     | Hidden Channels           | Dropout                  | Learning Rate                         | Number of eigenfunctions |
|-------------|---------------------------|--------------------------|---------------------------------------|--------------------------|
| Texas       | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Wisconsin   | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Cornell     | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Squirrel    | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Chameleon   | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Citeseer    | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Pubmed      | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |
| Cora        | [32,64,128,256,512, 1024] | [0, 0.1, 0.2, 0.3, 0.35] | [0.0001,0.001, 0.002, 0.01, 0.2 0.03] | [2,3,5,8,10,30]          |

