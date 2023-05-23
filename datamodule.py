import glob
import gzip
import os
import pickle

import cv2
# import matplotlib as mpl
# mpl.use('agg')  # Must be before pyplot import to avoid memory leak
# import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from world_model import WorldModel

def biternion_to_angle(x, y):
    '''Converts biternion tensor representation to positive angle tensor.
    Args:
        x: Biternion 'x' component of shape (batch_n, n, n)
        y: Biternion 'y' component of shape (batch_n, n, n)
    '''
    ang = torch.atan2(y, x)
    # Add 360 deg to negative angle elements
    mask = (ang < 0).float()
    ang = ang + 2.0*np.pi*mask
    return ang


def add_ang_to_dict(ang_dict, ang, i, j):
    if (i, j) not in ang_dict.keys():
        ang_dict[(i, j)] = []
    ang_dict[(i, j)].append(ang)
    return ang_dict


def get_angs_from_dict(ang_dict, i, j):
    return ang_dict[(i, j)]


class BEVDataset(Dataset):
    '''
    Returns (input_tensors, label_tensors) torch.Tensor.

        input_tensors[0] --> Road semantic 0|1
        input_tensors[1] --> Road intensity (0,1)

        label_tensors[0] --> Path label 0|1
        label_tensors[1] --> Directional x label (-1,1)
        label_tensors[2] --> Directional y label (-1,1)
    '''

    def __init__(self,
                 abs_root_path,
                 world_model,
                 do_rotation=False,
                 num_samples=8,
                 batch_size=8,
                 prob_use_pred=0.,
                 input_type='present',
                 get_gt_lanes=False,
                 rm_static_dyn_obj=False,
                 wm_conditioning='traj_full',
                 wm_temp=1.,
                 skip_wm=False):
        '''
        '''
        self.abs_root_path = abs_root_path
        self.sample_paths = glob.glob(
            os.path.join(self.abs_root_path, '*', '*.pkl.gz'))

        self.sample_paths = [
            os.path.relpath(path, self.abs_root_path)
            for path in self.sample_paths
        ]
        self.sample_paths.sort()

        self.num_samples = num_samples
        self.batch_size = batch_size

        self.world_model = world_model
        self.wm_conditioning = wm_conditioning
        self.wm_temp = wm_temp
        self.skip_wm = skip_wm

        self.do_rotation = do_rotation
        self.temp = 1.
        self.prob_use_pred = prob_use_pred
        self.input_type = input_type
        self.get_gt_lanes = get_gt_lanes
        self.rm_static_dyn_obj = rm_static_dyn_obj

        # Road marking intensity transformation
        self.int_scaler = 20
        self.int_sep_scaler = 20
        self.int_mid_threshold = 0.5

        # Representation size
        self.res = 256
        self.min_elements = 0.01 * 256 * 256

        self.static_dyn_obj_threshold = 0.5  # Range (0,1)

    def __len__(self):
        return len(self.sample_paths)

    def create_traj_label_set(self, trajs):
        # 1. (x,y) --> angle
        # 2. Add angle by img coord
        #####################################
        #  Create set of trajectory labels
        #####################################
        traj_labels = []
        dir_x_labels = []
        dir_y_labels = []
        for traj in trajs:

            # traj: (N, 2) matrix with (i, j) coordinates
            traj = traj[:, 0:2]
            traj[:, 1] = 255 - traj[:, 1]

            # Convert to point list
            n = traj.shape[0]
            traj = [(int(traj[idx, 0]), int(traj[idx, 1])) for idx in range(n)]

            traj = self.remove_duplicate_pnts(traj)

            # intensity = self.road_marking_transform(intensity)

            traj_label = self.draw_trajectory(traj, 256, 256, traj_width=1)
            dir_x_label, dir_y_label = self.draw_directional_trajectory(
                traj, 256, 256, traj_width=2)

            traj_labels.append(traj_label)
            dir_x_labels.append(dir_x_label)
            dir_y_labels.append(dir_y_label)

        ##################################################
        #  Merge labels to multimodal trajectory labels
        ##################################################
        # Add all labels into one
        traj_label_all = np.zeros((self.res, self.res))
        for traj_label in traj_labels:
            traj_label_all = np.logical_or(traj_label_all, traj_label)

        # Create a dict with list of angles indexed by coord.
        # ang_dict = {}
        max_elems = 256 * 256
        angs = -1 * np.ones((max_elems, 3))
        ang_idx = 0
        for idx in range(len(dir_x_labels)):
            dir_x_label = dir_x_labels[idx]
            dir_y_label = dir_y_labels[idx]

            mask = np.abs(dir_x_label) + np.abs(dir_y_label) > 0
            dir_ijs = np.argwhere(mask)  # (N,2)
            dir_ijs = dir_ijs.tolist()  # [(i,j), ... ]

            ang = biternion_to_angle(torch.tensor(dir_x_label),
                                     torch.tensor(dir_y_label))
            ang = ang.numpy()

            for dir_ij in dir_ijs:
                i = dir_ij[0]
                j = dir_ij[1]
                # ang_dict = add_ang_to_dict(ang_dict, ang[i, j].item(), i, j)
                # angs.append((i, j, ang[i, j]))
                angs[ang_idx, 0] = i
                angs[ang_idx, 1] = j
                angs[ang_idx, 2] = ang[i, j]
                ang_idx += 1
                if ang_idx > max_elems:
                    raise Exception('Directional elements over limit')

        return traj_label_all, angs

    def process_sample(self, sample: dict):
        '''
        Args:
            sample:

        Returns:
            x: VDVAE posterior matching input tensor(6,256,256) in [0, 1]
            dynamic: (256,256) in [0, 1]
            traj_label_all: (256,256) in {0, 1}
            ang_dict: key (i,j) --> value 'ang' in rad
        '''
        if self.input_type == 'full':
            road = sample['road_full'].astype(np.float32)
            intensity = sample['intensity_full'].astype(np.float32)
            rgb = sample['rgb_full'].astype(np.float32)
            dynamic = sample['dynamic_full'].astype(np.float32)
            # trajs = sample['trajs_full']
        elif self.input_type == 'present':
            road = sample['road_present'].astype(np.float32)
            intensity = sample['intensity_present'].astype(np.float32)
            rgb = sample['rgb_present'].astype(np.float32)
            dynamic = sample['dynamic_present'].astype(np.float32)
            # trajs = sample['trajs_present']
        elif self.input_type == 'future':
            road = sample['road_future'].astype(np.float32)
            intensity = sample['intensity_future'].astype(np.float32)
            rgb = sample['rgb_future'].astype(np.float32)
            dynamic = sample['dynamic_future'].astype(np.float32)
            # trajs = sample['trajs_future']
        else:
            raise IOError(f'Undefined type ({self.input_type})')

        trajs_present = sample['trajs_present']
        trajs_full = sample['trajs_full']
        gt_lanes = sample['gt_lanes']

        traj_label_present, angs_present = self.create_traj_label_set(
            trajs_present)
        traj_label_full, angs_full = self.create_traj_label_set(trajs_full)

        # Make non-road intensity 0.5 (i.e. unknown)
        road_mask = road > 0.5
        intensity[~road_mask] = 0.5

        # Remove static dynamic objects from observations
        if self.rm_static_dyn_obj:
            mask = dynamic > self.static_dyn_obj_threshold
            road[mask] = 1.
            intensity[mask] = 0.5
            mask_3ch = np.tile(np.expand_dims(mask, 0), (3, 1, 1))
            rgb[mask_3ch] = np.zeros_like(rgb)[mask_3ch]

        obs_mask = ~(road == 0.5)

        # VDVAE posterior matching input tensor 'x' (6, H, W)
        x = np.concatenate([
            np.expand_dims(road, 0),
            np.expand_dims(intensity, 0),
            rgb,
            np.expand_dims(obs_mask, 0),
        ])

        x = torch.tensor(x)
        dynamic = torch.tensor(dynamic)
        traj_label_present = torch.tensor(traj_label_present)
        ang_present = torch.tensor(angs_present)
        traj_label_full = torch.tensor(traj_label_full)
        ang_full = torch.tensor(angs_full)
        sample = [
            x, dynamic, traj_label_present, ang_present, traj_label_full,
            ang_full
        ]
        if self.get_gt_lanes:
            gt_lanes_label, angs_gt_lanes = self.create_traj_label_set(
                gt_lanes)
            gt_lanes_label = torch.tensor(gt_lanes_label)
            angs_gt_lanes = torch.tensor(angs_gt_lanes)
            sample += [gt_lanes_label, angs_gt_lanes]

        return sample

    def __getitem__(self, idx):
        while True:
            sample_path = self.sample_paths[idx]
            sample_path = os.path.join(self.abs_root_path, sample_path)

            sample = self.read_compressed_pickle(sample_path)

            num_obs_elem_present = np.sum(sample['road_present'] != 0.5)
            num_obs_elem_future = np.sum(sample['road_future'] != 0.5)
            num_obs_elem_full = np.sum(sample['road_full'] != 0.5)
            if (num_obs_elem_present < self.min_elements
                    or num_obs_elem_future < self.min_elements
                    or num_obs_elem_full < self.min_elements):
                idx = self.get_random_sample_idx()
            else:
                break

        out = self.process_sample(sample)
        x = out[0]
        dynamic = out[1]
        traj_label_present = out[2]
        angs_present = out[3]
        traj_label_full = out[4]
        angs_full = out[5]
        if self.get_gt_lanes:
            gt_lanes_label = out[6]
            angs_gt_lanes = out[7]

        # Future observed trajectory conditioning
        x_cond = x.clone()
        if self.wm_conditioning == 'traj_full':
            mask = traj_label_full == 1
        elif self.wm_conditioning == 'gt_lanes':
            mask = gt_lanes_label == 1
        elif self.wm_conditioning is None:
            mask = torch.zeros_like(traj_label_full)
        else:
            raise IOError('Undefined world model conditioning')
        # TODO Make mask bigger by dillation (model may ignore conditioning)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.numpy().astype(float), kernel)
        mask = torch.tensor(mask, dtype=torch.bool)
        x_cond[0][mask] = 1

        ################
        #  Completion
        ################
        if not self.skip_wm:
            # Transform value range [0,1] --> [-1,1] (except obs mask)
            x_cond[:5] = 2 * x_cond[:5] - 1

            # TODO Add trajectory all as guidance for prediction
            x_hat = self.world_model.sat_sample(x_cond,
                                                traj_label_full,
                                                self.batch_size,
                                                temp=self.wm_temp)
        else:
            x_hat = x_cond

        label = {
            # NOTE Sample should be generated as dynamic yes|no
            'dynamic': dynamic,
            'traj_present': traj_label_present,
            'angs_present': angs_present,
            'traj_full': traj_label_full,
            'angs_full': angs_full,
            'scene_idx': sample['scene_idx'],
            'map': sample['map'],
            'ego_global_x': sample['ego_global_x'],
            'ego_global_y': sample['ego_global_y'],
        }

        if self.get_gt_lanes:
            label.update({
                'gt_lanes': gt_lanes_label,
                'gt_angs': angs_gt_lanes
            })

        return x, x_hat, label

    def get_random_sample_idx(self):
        return np.random.randint(0, self.num_samples)

    def road_marking_transform(self, intensity_map):
        '''
        Args:
            intensity_map: Value interval (0, 1)
        '''
        intensity_map = self.int_scaler * self.sigmoid(
            self.int_sep_scaler * (intensity_map - self.int_mid_threshold))
        # Normalization
        intensity_map[intensity_map > 1.] = 1.
        return intensity_map

    def draw_trajectory(self,
                        traj: np.ndarray,
                        I: int,
                        J: int,
                        traj_width: int = 5):
        '''
        Args:
            traj: (N,2) matrix with (i, j) coordinates
        '''
        label = np.zeros((I, J))
        for idx in range(len(traj) - 1):

            pnt_0 = traj[idx]
            pnt_1 = traj[idx + 1]

            # pnt_0 = poses[idx].astype(int)
            # pnt_1 = poses[idx + 1].astype(int)
            # pnt_0 = tuple(pnt_0)
            # pnt_1 = tuple(pnt_1)

            cv2.line(label, pnt_0, pnt_1, 1, traj_width)

        return label

    def draw_directional_trajectory(self, traj, I, J, traj_width=5):
        '''
        '''
        circle_radius = int(np.ceil(0.5 * (traj_width + 1)))

        label_x = np.zeros((I, J))
        label_y = np.zeros((I, J))
        for idx in range(len(traj) - 1):

            pnt_0 = traj[idx]
            pnt_1 = traj[idx + 1]

            if pnt_0 == pnt_1:
                continue

            vec_x, vec_y = self.cal_norm_vector(pnt_0, pnt_1)

            cv2.line(label_x, pnt_0, pnt_1, vec_x, traj_width)
            cv2.line(label_y, pnt_0, pnt_1, vec_y, traj_width)

        # Average mid-points
        for idx in range(1, len(traj) - 1):
            pnt_0 = traj[idx - 1]
            pnt_1 = traj[idx]
            pnt_2 = traj[idx + 1]
            # Calculate the normalized average vector between two
            vec_x_before, vec_y_before = self.cal_norm_vector(pnt_0, pnt_1)
            vec_x_after, vec_y_after = self.cal_norm_vector(pnt_1, pnt_2)
            vec_x_avg = (vec_x_before + vec_x_after)
            vec_y_avg = (vec_y_before + vec_y_after)
            vec_avg_len = np.sqrt(vec_x_avg**2 + vec_y_avg**2)

            if vec_avg_len < 1e-9:
                continue

            vec_x_avg = vec_x_avg / vec_avg_len
            vec_y_avg = vec_y_avg / vec_avg_len

            cv2.circle(label_x, pnt_1, circle_radius, vec_x_avg, -1)
            cv2.circle(label_y, pnt_1, circle_radius, vec_y_avg, -1)

        return label_x, label_y

    @staticmethod
    def remove_duplicate_pnts(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    @staticmethod
    def cal_norm_vector(pnt_0, pnt_1):
        dx = pnt_1[0] - pnt_0[0]
        dy = pnt_1[1] - pnt_0[1]

        length = np.sqrt(dx**2 + dy**2)
        vec_x = dx / length
        vec_y = -dy / length  # NOTE Inverted y-axis!

        if (length == 0):
            print(pnt_0)
            print(pnt_1)
            print(dx)
            print(dy)
            print(length)
            print(vec_x)
            print(vec_y)

        return vec_x, vec_y

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def read_compressed_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)


# class BEVDummyDataset(BEVDataset):
#
#     def __init__(self):
#         pass


class BEVDataModule(pl.LightningDataModule):

    def __init__(self,
                 world_model,
                 train_data_dir: str = "./",
                 val_data_dir: str = "./",
                 batch_size: int = 128,
                 num_samples: int = 8,
                 num_workers: int = 0,
                 persistent_workers=False,
                 do_rotation: bool = False,
                 prob_use_pred: float = 0.,
                 input_type: str = 'present',
                 get_gt_lanes: bool = False,
                 rm_static_dyn_obj: bool = False,
                 wm_conditioning: str = 'traj_full',
                 wm_temp: float = 1.,
                 skip_wm: bool = False):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.bev_dataset_train = BEVDataset(
            self.train_data_dir,
            world_model,
            do_rotation=do_rotation,
            num_samples=num_samples,
            batch_size=batch_size,
            prob_use_pred=prob_use_pred,
            input_type=input_type,
            get_gt_lanes=get_gt_lanes,
            rm_static_dyn_obj=rm_static_dyn_obj,
            wm_conditioning=wm_conditioning,
            wm_temp=wm_temp,
            skip_wm=skip_wm)
        self.bev_dataset_val = BEVDataset(self.val_data_dir,
                                          world_model,
                                          num_samples=num_samples,
                                          batch_size=batch_size,
                                          prob_use_pred=prob_use_pred,
                                          input_type=input_type,
                                          get_gt_lanes=get_gt_lanes,
                                          rm_static_dyn_obj=rm_static_dyn_obj,
                                          wm_conditioning=wm_conditioning,
                                          wm_temp=wm_temp,
                                          skip_wm=skip_wm)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.bev_dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )


def write_compressed_pickle(obj, filename, write_dir):
    '''Converts an object into byte representation and writes a compressed file.
    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    path = os.path.join(write_dir, f"{filename}.gz")
    pkl_obj = pickle.dumps(obj)
    try:
        with gzip.open(path, "wb") as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)


if __name__ == '__main__':
    '''
    For visualizing dataset tensors.

    NOTE Needs to be run with VDVAE input arguments.
    '''
    # import matplotlib.pyplot as plt

    from viz.viz_dataset import viz_dataset_sample

    # Load model as global variable to avoid duplicate loadings by 'train' and
    # 'val' dataloader
    world_model = WorldModel()

    batch_size = 1
    num_samples = 8
    prob_use_pred = 0.
    input_type = 'present'
    get_gt_lanes = True
    rm_static_dyn_obj = True
    wm_conditioning = None  # 'gt_lanes'
    wm_temp = 0.6  # Better to keep temp < 1. ?
    skip_wm = True

    ###################################
    #  Generate preprocessed dataset
    ###################################

    bev = BEVDataModule(
        world_model,
        '/media/robin/Drive2/bev_nuscenes_256px_v01_boston_seaport_gt_huracan',
        '/media/robin/Drive2/bev_nuscenes_256px_v01_boston_seaport_gt_huracan',
        # '/home/robin/projects/pc-accumulation-lib/bev_kitti360_256px_aug_gt_3_rev',
        # '/home/robin/projects/pc-accumulation-lib/bev_kitti360_256px_aug_gt_3_rev',
        # '/home/robin/projects/vdvae/bev_kitti360_256px_aug_gt_3_rev_preproc_val_completed',
        # '/home/robin/projects/vdvae/bev_kitti360_256px_aug_gt_3_rev_preproc_val_completed',
        batch_size,
        num_samples,
        prob_use_pred=prob_use_pred,
        input_type=input_type,
        get_gt_lanes=get_gt_lanes,
        rm_static_dyn_obj=rm_static_dyn_obj,
        wm_conditioning=wm_conditioning,
        wm_temp=wm_temp,
        skip_wm=skip_wm)

    dataloader = bev.train_dataloader(shuffle=False)

    num_samples = len(bev.bev_dataset_train)

    bev_idx = 0
    subdir_idx = 0
    savedir = '/media/robin/Drive2/bev_nuscenes_256px_v01_boston_seaport_gt_huracan_raw_preproc'

    # Three process example
    #     num_processes = 3
    #     process_idx = 0|1|2
    num_processes = 1
    process_idx = 0

    for idx, batch in enumerate(dataloader):

        x, x_hat, label = batch

        # Remove batch index (B,C,H,W) --> (C,H,W)
        x = x[0]
        x_hat = x_hat[0]
        for key in label.keys():
            label[key] = label[key][0]

        if bev_idx >= 1000:
            bev_idx = 0
            subdir_idx += 1
        output_path = f'{savedir}/subdir{subdir_idx:03d}'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        bev_idx_str = str(bev_idx).zfill(3)
        filename = f'bev_{bev_idx_str}.pkl'
        sample = (x_hat, label)
        write_compressed_pickle(sample, filename, output_path)

        filepath = os.path.join(output_path, f'viz_{bev_idx_str}.png')
        viz_dataset_sample(x, x_hat, label, filepath, get_gt_lanes)

        bev_idx += 1

        # if idx % 100 == 0:
        print(f'idx {idx} / {num_samples} ({idx/num_samples*100:.2f}%)')
