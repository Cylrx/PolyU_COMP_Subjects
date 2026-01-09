# DSAI4204 Data Mining and Data WareHousing: Group Project

## Get Started

```
conda create -n hpp python=3.12
conda activate hpp

# ensure run the following from project root:
poetry install
poetry run python main.py
```

Three different modes: 

- **Grid Mode**: for automaticallly running numerous experiments by permutating the `run_grid` parameter in `config.yaml`. Enable this mode by setting `grid_mode=True`.
- **Default Mode**: runs a single experiment using the `run_overlays` parameter in `config.yaml`. Enable this mode by setting `grid_mode=False`.
- **Eval Mode**: loads a pretrained model from the model folder and evaluate it on a holdout dataset. This requires manually creating a holdout split and is mainly used for hyperparameter tuning in dynamic stacking. 