# import argparse
import gzip
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from datamodule import BEVDataset
from world_model import WorldModel


def read_compressed_pickle(path):
    try:
        with gzip.open(path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


def create_viz(idx, x, x_pred, x_full):

    # Normalize values (-1,1) --> (0,1)
    x = 0.5 * (x + 1)

    # Mask out non-road intensity to 0.5
    mask = x[0] == 0
    x[1][mask] = 0.5

    mask = x_pred[0] == 0
    x_pred[1][mask] = 0.5

    mask = x_full[0] == 0.5
    x_full[1][mask] = 0.5

    row_1 = np.concatenate([x[0], x[1]], axis=1)
    row_2 = np.concatenate([x_pred[0], x_pred[1]], axis=1)
    row_3 = np.concatenate([x_full[0], x_full[1]], axis=1)

    img = np.concatenate([row_1, row_2, row_3], axis=0)

    cm = plt.get_cmap('viridis')
    img = cm(img)
    img = img[:, :, :3]  # Remove alpha channel

    col = np.concatenate([x[2:5], x_pred[2:5], x_full[2:5]], axis=1)
    col = np.transpose(col, (1, 2, 0))

    img = np.concatenate([img, col], axis=1)

    size_per_fig = 6
    cols = 3
    rows = 3
    plt.figure(figsize=(cols * size_per_fig, rows * size_per_fig))

    plt.imshow(img, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{idx}.png'))
    plt.close()


def remove_duplicate_pnts(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


sample_path = 'test_sample_nuscenes.pkl.gz'
num_samples = 32
batch_size = 1
output_dir = 'out_dir'
temp = 1

world_model = WorldModel()

sample = read_compressed_pickle(sample_path)

if os.path.isdir(output_dir) is False:
    os.makedirs(output_dir)

dataset = BEVDataset('dummy_path', world_model)
dataset_full = BEVDataset('dummy_path', world_model, input_type='full')
out = dataset.process_sample(sample)
x = out[0]
x[:5] = 2 * x[:5] - 1

out_full = dataset_full.process_sample(sample)
x_full = out_full[0]

x_hats = []
num_inferences = num_samples // batch_size
if num_samples % batch_size > 0:
    num_inferences += 1
for sampling_idx in range(num_inferences):

    # Sample without caring about trajectory
    x_hat = world_model.sample(
        x,
        batch_size,
        temp,
    )
    for batch_idx in range(batch_size):
        x_hats.append(x_hat[batch_idx])
x_hats = x_hats[:num_samples]

for idx, x_hat in enumerate(x_hats):
    create_viz(idx, x, x_hat, x_full)
