# Predictive World Models from Real-World Partial Observations
Code accompanying the paper "Predictive World Models from Real-World Partial Observations" (IEEE MOST 2023) :tada: **_Best paper award_** :tada:

Paper link: [Predictive World Models from Real-World Partial Observations](https://arxiv.org/abs/2301.04783)

Video presentation link: TODO

Shared public data (incl. pretrained models): [Google Drive directory](https://drive.google.com/drive/folders/1bU6W0yeEz7TttEhS3Y3oDJgvtTd_9Oqu?usp=share_link)

![Predictive world model inference](https://github.com/robin-karlsson0/predictive-world-models/assets/34254153/5be007a3-217f-48a2-8f29-3e8f6bc91623)

# Installation

Download all submodules

```
git submodule update --init --recursive
```

The submodules are used for the following tasks
1. `pc-accumulation-lib`: Semantic point cloud accumulation library for generating partial observation BEV representations.
2. `lat_var_bev_pred_model`: Code for generating complete pseudo-GT representations for training the predictive world model.
3. `vdvae`: Code for implementing the predictive world model. Fork of the original VDVAE repository modified to a dual encoder posterior matching HVAE model.

The paper can be reproduced by generating data and training models using the code provided within this repository including all submodules.

Install dependencies

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install opencv-contrib-python
pip install pytorch-lightning
pip install scipy
```

Install VDVAE dependencies by following the README instructions inside the VDVAE directory.

```
export PYTHONPATH=/PATH/predictive-world-models/vdvae
```

while replacing `PATH` with the absolute path to the `predictive-world-models` directory.

# Run model

## NuScenes example

Download trained model and test sample by running the download script or manually from the project Google Drive directory.

```
sh download_model_nuscenes.sh
```

https://drive.google.com/drive/folders/1bU6W0yeEz7TttEhS3Y3oDJgvtTd_9Oqu?usp=sharing

The following files will be placed in the project root directory

```
predictive-world-models/pred_wm_model_ema_nuscenes.th
predictive-world-models/pred_wm_model_nuscenes.th
predictive-world-models/test_sample_nuscenes.pkl.gz
predictive-world-models/test_sample_nuscenes.png
```

Sample 36 plausible worlds based on the partially observable world represented by the test sample by running the script.

```
sh sample_worlds_nuscenes.sh
```

A set of sampled plausible worlds will be visualized in the output directory (`out_dir` by default).

![Predictive world model output on NuScenes](https://github.com/robin-karlsson0/predictive-world-models/assets/34254153/f26adafd-5f08-4ff7-82b7-a12381b68e90)

# File structure

```
predictive-world-models
|
└───lat_var_bev_pred_model/     # Pseudo-GT sample generation
|   └─── ...
|
└───pc-accumulation-lib/        # Observation accumulation framework
|   └─── ...
|
└───vdvae/                      # Predictive world model implementation
|   └─── ...
|
|   datamodule.py               # Reads and pre-processes input samples
|   download_model_nuscenes.sh  # Downloads model and test sample files
|   sample_worlds_nuscenes.py   # Runs the world model and save visualizations to disk
|   sample_worlds_nuscenes.sh   # Script including required environment variables
|   world_model.py              # World model inference interface
```