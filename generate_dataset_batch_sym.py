import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # environment settings
    tag = 'mixed'
    mark = 'sym'

    # load pickle files
    # use tqdm to show progress bar
    all_train_files = []
    all_eval_files = []

    for num_obj in range(1, 2):
        file_list = glob.glob(f"datasets/{tag}/trajs/*{num_obj}_objs.pkl")
        random.shuffle(file_list)
        # split the files into training and evaluation set
        split = int(0.95 * len(file_list))
        all_train_files += file_list[:split]
        all_eval_files += file_list[split:]
    
    def process(files, flag, batch_size):
        all_scenes_images = []
        all_success_grasps = []
        all_failure_grasps = []
        all_num_objs = []
        current_size = 0
        batch_idx = 0
        for pkl_file in tqdm.tqdm(files):
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
                # print(f"Processing {pkl_file_name}", current_size, len(data['samples']['success_indices']), f"batch {batch_idx}")
                success = data['samples']['success_indices'][i]
                # randomly select a failure grasp
                failure = random.choice(data['samples']['failure_indices'])

                success_grasp = data['samples']['grasp_poses'][success]
                failure_grasp = data['samples']['grasp_poses'][failure]
                    
                # create pairs
                all_scenes_images.append(scene_image)
                all_success_grasps.append(success_grasp)
                all_failure_grasps.append(failure_grasp)
                all_num_objs.append(int(pkl_file_name.split('_')[2]))
                current_size += 1

                if current_size == batch_size:
                    # save the batch
                    with h5py.File(f"datasets/{tag}/processed/{flag}/{mark}_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
                        hf.create_dataset(f'scenes', data=all_scenes_images)
                        hf.create_dataset(f'success_grasps', data=all_success_grasps)
                        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
                        hf.create_dataset(f'num_objs', data=all_num_objs)

                    # reset the data
                    all_scenes_images = []
                    all_success_grasps = []
                    all_failure_grasps = []
                    all_num_objs = []
                    current_size = 0
                    batch_idx += 1

                    print(f"Saved {flag} suc batch {batch_idx}")
            
            for i in range(len(data['samples']['failure_indices'])):
                # print(f"Processing {pkl_file_name}", current_size, len(data['samples']['success_indices']), f"batch {batch_idx}")
                success = random.choice(data['samples']['success_indices'])
                # randomly select a failure grasp
                failure = data['samples']['failure_indices'][i]

                success_grasp = data['samples']['grasp_poses'][success]
                failure_grasp = data['samples']['grasp_poses'][failure]
                    
                # create pairs
                all_scenes_images.append(scene_image)
                all_success_grasps.append(success_grasp)
                all_failure_grasps.append(failure_grasp)
                all_num_objs.append(int(pkl_file_name.split('_')[2]))
                current_size += 1

                if current_size == batch_size:
                    # save the batch
                    with h5py.File(f"datasets/{tag}/processed/{flag}/{mark}_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
                        hf.create_dataset(f'scenes', data=all_scenes_images)
                        hf.create_dataset(f'success_grasps', data=all_success_grasps)
                        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
                        hf.create_dataset(f'num_objs', data=all_num_objs)

                    # reset the data
                    all_scenes_images = []
                    all_success_grasps = []
                    all_failure_grasps = []
                    all_num_objs = []
                    current_size = 0
                    batch_idx += 1

                    print(f"Saved {flag} fal batch {batch_idx}")

        # save the last batch
        if current_size > 0:
            batch_idx += 1
            with h5py.File(f"datasets/{tag}/processed/{flag}/{mark}_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
                hf.create_dataset(f'scenes', data=all_scenes_images)
                hf.create_dataset(f'success_grasps', data=all_success_grasps)
                hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
                hf.create_dataset(f'num_objs', data=all_num_objs)

            print(f"Saved {flag} final batch {batch_idx}")

    process(all_train_files, 'train', batch_size=1024)
    process(all_eval_files, 'eval', batch_size=2048)