# MaxClique

Code for the paper [Learning to Rank: How GNNs Solve Max-Clique and Sparse PCA](https://ojs.aaai.org/index.php/AAAI/article/view/39734)

The contribution to this codebase is with equal contribution from Omri Haber and Elad Shoham (regardless of what the commit history says)

## Train

### Scattering Model

Instructions for training the scattering model:

1. Open the `train_scattering.py` file.
2. Edit the parameters you want to train with.
3. Run the training script:
   ```bash
   python train_scattering.py
   ```

### Erdos Model

Instructions for training the Erdos model:

1. Open the `train_erdos.py` file.
2. Edit the dataset you want to train on, and/or the parameters you wish to edit.
3. Run the training script:
   ```bash
   python train_erdos.py
   ```

## Evaluate

Instructions for evaluating each file:

1. Navigate to the `evaluation` directory.
2. Run the evaluation script for the desired model:
   ```bash
   python evaluate.py --model scattering --data data/scattering_data.json
   python evaluate.py --model erdos --data data/erdos_data.json
   ```

## DevContainers

Using VSCode devcontainers theres a `Dockerfile` and `devcontainer.json` set under the `.devcontainer` folder

## Install locally

1. Install core requirements by running: `pip3 install -r requirements.txt`

2. Install torch-sparse and torch-scatter which require torch to be installed by running: `pip3 install -r requirements-post.txt`

# Datasets Generation

## Format

We use the `.jsonl` format to serialize the datasets aswell as the results of our experiments.

## PMC(clique solver)

We use the following [solver](https://github.com/ryanrossi/pmc) inorder to use it you must compile it and place the required files under the `pmc` folder.

## Planted/Random

1. Open `generate_difficult_instances.py`
2. Edit parameters(size and difficulty) in the main section.
3. Run `python generate_difficult_instances.py`

## TUDataset(COLLAB/IMDB)

1. Open `TUDataset2JSONL.py`
2. Under `__main__` edit the desired dataset and output directory
3. Run `python TUDataset2JSONL.py`

## TWITTER

The raw dataset can be downloaded with from
[here](https://snap.stanford.edu/data/ego-Twitter.html).
After extracting its content run the following to generate the `.jsonl` dataset file:

`python generate_twitter.py`

# Datasets(download)

[N=500 Easy](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n500_easy_instances.jsonl)
[N=500 Medium](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n500_medium_instances.jsonl)
[N=500 Hard](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n500_hard_instances.jsonl)

[N=1000 Easy](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n1000_easy_instances.jsonl)
[N=1000 Medium](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n1000_medium_instances.jsonl)
[N=1000 Hard](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/n1000_hard_instances.jsonl)

[Orkut](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/com-orkut.jsonl)
[Twitter](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/twitter.jsonl)
[Youtube](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/com-youtube.jsonl)
[Facebook](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/facebook.jsonl)
[IMDB](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/imdb_binary.jsonl)
[Collab](https://s3.eu-central-1.wasabisys.com/clique/data/with_clique/collab.jsonl)

# Code used/based from the following projects and papers:

- [pmc](https://github.com/ryanrossi/pmc) - Solver used to find Max-Clique
- [Geometric **Scattering** Maximal Clique (SNN)](https://github.com/yimengmin/GeometricScatteringMaximalClique) - One of the models used in the research 
- [**Erdos** Goes Neural (ENN)](https://github.com/Stalence/erdos_neu) - One of the models used in the research 
