import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# preprocess the trajs
if __name__ == '__main__':
    # environment settings
    tag = 'single'
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []

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

        # randomly select one successful grasp and one failure grasp
        success_grasps = success_grasps[np.random.choice(np.arange(len(success_grasps)), 1)][0]
        failure_grasps = failure_grasps[np.random.choice(np.arange(len(failure_grasps)), 1)][0]

        all_scenes_images.append(scene_image)
        all_success_grasps.append(success_grasps)
        all_failure_grasps.append(failure_grasps)

    all_scenes_images = np.array(all_scenes_images)
    all_success_grasps = np.array(all_success_grasps)
    all_failure_grasps = np.array(all_failure_grasps)

    print(all_scenes_images.shape, all_success_grasps.shape, all_failure_grasps.shape)

    # split into training and evaluation datasets
    train_all_scenes_images, test_all_scenes_images, train_all_success_grasps, test_all_success_grasps, train_all_failure_grasps, test_all_failure_grasps = \
        train_test_split(all_scenes_images, all_success_grasps, all_failure_grasps, test_size=0.1, random_state=42)

    with h5py.File(f"datasets/{tag}/clip_dataset_100k_train.h4", 'w') as f:
        f.create_dataset('scenes', data=train_all_scenes_images)
        f.create_dataset('success_grasps', data=train_all_success_grasps)
        f.create_dataset('failure_grasps', data=train_all_failure_grasps)
        f.close()

    with h5py.File(f"datasets/{tag}/clip_dataset_100k_eval.h5", 'w') as f:
        f.create_dataset('scenes', data=test_all_scenes_images)
        f.create_dataset('success_grasps', data=test_all_success_grasps)
        f.create_dataset('failure_grasps', data=test_all_failure_grasps)
        f.close()
