import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np
from multiprocessing import Pool

def process_files(file_list, tag, batch_size, process_idx):
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []
    current_size = 0
    batch_idx = 0

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
            for j in range(len(data['samples']['failure_indices'])):
                success = data['samples']['success_indices'][i]
                failure = data['samples']['failure_indices'][j]

                success_grasp = data['samples']['grasp_poses'][success]
                failure_grasp = data['samples']['grasp_poses'][failure]

                # create pairs
                all_scenes_images.append(scene_image)
                all_success_grasps.append(success_grasp)
                all_failure_grasps.append(failure_grasp)

                current_size += 1

                if current_size == batch_size:
                    batch_idx += 1
                    # save the dataset
                    with h5py.File(f"datasets/{tag}/processed/bs_{batch_size}_no_{process_idx}_{batch_idx}.h5", 'a') as hf:
                        hf.create_dataset(f'scenes', data=all_scenes_images)
                        hf.create_dataset(f'success_grasps', data=all_success_grasps)
                        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)

                    # reset the data
                    all_scenes_images = []
                    all_success_grasps = []
                    all_failure_grasps = []
                    current_size = 0

                    print(f"Process {process_idx} saved batch {batch_idx}")

    # Save remaining data if any
    if current_size > 0:
        batch_idx += 1
        with h5py.File(f"datasets/{tag}/processed/bs_{batch_size}_no_{process_idx}_{batch_idx}.h5", 'a') as hf:
            hf.create_dataset(f'scenes', data=all_scenes_images)
            hf.create_dataset(f'success_grasps', data=all_success_grasps)
            hf.create_dataset(f'failure_grasps', data=all_failure_grasps)

        print(f"Process {process_idx} saved batch {batch_idx}, final batch length: {current_size}")

if __name__ == '__main__':
    # environment settings
    tag = 'mixed'
    num_obj = 1
    batch_size = 2048

    # load pickle files
    file_list = glob.glob(f"datasets/{tag}/trajs/*objs.pkl")
    random.shuffle(file_list)

    # Number of processes
    num_processes = 50

    # Split file list into sublists for each process
    file_lists = np.array_split(file_list, num_processes)

    # Use multiprocessing to process files
    with Pool(num_processes) as pool:
        for i, sublist in enumerate(file_lists):
            pool.apply_async(process_files, args=(sublist, tag, batch_size, i))

        pool.close()
        pool.join()

    print("All processes completed.")
