import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np


if __name__ == '__main__':
    # environment settings
    tag = 'mixed'

    # save all data
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []

    # load pickle files
    # use tqdm to show progress bar
    file_list = glob.glob(f"datasets/{tag}/trajs/*objs.pkl")
    random.shuffle(file_list)

    for pkl_file in tqdm.tqdm(file_list):
        pkl_file_name = os.path.basename(pkl_file).split('.')[0]
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # load masks and depth images
        masks = np.load(f"datasets/{tag}/masks/{pkl_file_name}_masks.npz")
        depth_image = data['samples']['depth_image']
        seg_mask = masks['seg_mask']
        heightmap = masks['heightmap']
        # get scene fusion images
        scene_image = np.stack([seg_mask, depth_image, heightmap])

        for i in range(len(data['samples']['success_indices'])):
            success = data['samples']['success_indices'][i]
            # randomly select a failure grasp
            failure = random.choice(data['samples']['failure_indices'])

            success_grasp = data['samples']['grasp_poses'][success]
            failure_grasp = data['samples']['grasp_poses'][failure]
                
            # create pairs
            all_scenes_images.append(scene_image)
            all_success_grasps.append(success_grasp)
            all_failure_grasps.append(failure_grasp)

    with h5py.File(f"datasets/{tag}/processed/processed_total.h5", 'a') as hf:
        hf.create_dataset(f'scenes', data=all_scenes_images)
        hf.create_dataset(f'success_grasps', data=all_success_grasps)
        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
        
        