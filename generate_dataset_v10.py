import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np

import torch

def statistical_features(grasp_vectors):
    """
    grasp_vectors: Tensor of shape (num_grasps, grasp_dim)
    returns: Tensor of shape (num_features * grasp_dim,)
    """
    mean = np.mean(grasp_vectors, axis=0)
    max_val = np.max(grasp_vectors, axis=0)
    min_val = np.min(grasp_vectors, axis=0)
    std = np.std(grasp_vectors, axis=0)

    return np.concatenate([mean, max_val, min_val, std])


# preprocess the trajs
if __name__ == '__main__':
    # environment settings
    tag = 'single'
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []
    patch_size = 20

    # all_sg_len = []
    # all_fg_len = []

    file_list = glob.glob(f"datasets/{tag}/trajs/*_objs.pkl")
    random.shuffle(file_list)
    for pkl_file in tqdm.tqdm(file_list):
        pkl_file_name = os.path.basename(pkl_file).split('.')[0]
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        num_objs = int(pkl_file_name.split('_')[2])

        # load masks and depth images
        masks = np.load(f"datasets/{tag}/masks/{pkl_file_name}_masks.npz")
        depth_image = data['samples']['depth_image']
        seg_mask = masks['seg_mask']
        heightmap = masks['heightmap']
        # get scene fusion images
        scene_image = np.stack([seg_mask, depth_image, heightmap])

        grasp_poses = np.array(data['samples']['grasp_poses'])
        success_grasps = grasp_poses[data['samples']['success_indices']]
        failure_grasps = grasp_poses[data['samples']['failure_indices']]
        
        sg_len = len(success_grasps)
        fg_len = len(failure_grasps)

        if len(success_grasps) > patch_size:
            # randomly sample patch_size successful grasps
            success_grasps = success_grasps[np.random.choice(np.arange(sg_len), patch_size, replace=False)]
        else:
            num_missing = patch_size - len(success_grasps)
            # randomly sample from failure grasps
            for _ in range(num_missing):
                patch = success_grasps[np.random.choice(np.arange(sg_len), 1, replace=False)]
                success_grasps = np.concatenate([success_grasps, patch], axis=0)

        if len(failure_grasps) > patch_size:
            # randomly sample patch_size successful grasps
            failure_grasps = failure_grasps[np.random.choice(np.arange(fg_len), patch_size, replace=False)]
        else:
            num_missing = patch_size - len(failure_grasps)
            # randomly sample from failure grasps
            for _ in range(num_missing):
                patch = failure_grasps[np.random.choice(np.arange(fg_len), 1, replace=False)]
                failure_grasps = np.concatenate([failure_grasps, patch], axis=0)

        all_scenes_images.append(scene_image)
        all_success_grasps.append(success_grasps)
        all_failure_grasps.append(failure_grasps)

    all_scenes_images = np.array(all_scenes_images)
    all_success_grasps = np.array(all_success_grasps)
    all_failure_grasps = np.array(all_failure_grasps)

    print(all_scenes_images.shape, all_success_grasps.shape, all_failure_grasps.shape)

    with h5py.File(f"datasets/{tag}/clip_dataset_patch.h5", 'w') as f:
        f.create_dataset('scenes', data=all_scenes_images)
        f.create_dataset('success_grasps', data=all_success_grasps)
        f.create_dataset('failure_grasps', data=all_failure_grasps)
        f.close()