import os
import glob
import tqdm
import h5py
import random
import pickle
import numpy as np

# preprocess the trajs
if __name__ == '__main__':
    # environment settings
    tag = 'mixed'
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []
    all_num_objs = []
    current_size = 0
    batch_idx = 0
    batch_size = 512    

    # use tqdm to show progress bar
    for j in range(1, 2):
        # load all pickle files
        file_list = glob.glob(f"datasets/{tag}/trajs/*{j}_objs.pkl")
        random.shuffle(file_list)

        # get 100 files for evaluation
        eval_files = random.sample(file_list, 100)
        eval_scenes_images = []
        eval_success_grasps = []
        eval_failure_grasps = []
        eval_num_objs = []

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

            for i in range(len(data['samples']['success_indices'])):
                success = data['samples']['success_indices'][i]
                # randomly select a failure grasp
                failure = random.choice(data['samples']['failure_indices'])
                success_grasp = data['samples']['grasp_poses'][success]
                failure_grasp = data['samples']['grasp_poses'][failure]

                # save the first pair for evaluation
                if pkl_file in eval_files and i == 0:
                    eval_scenes_images.append(scene_image)
                    eval_success_grasps.append(success_grasp)
                    eval_failure_grasps.append(failure_grasp)
                    eval_num_objs.append(num_objs)
                    continue
                    
                # create pairs
                all_scenes_images.append(scene_image)
                all_success_grasps.append(success_grasp)
                all_failure_grasps.append(failure_grasp)
                all_num_objs.append(num_objs)
                current_size += 1

                if current_size == batch_size:
                    # save the batch
                    with h5py.File(f"datasets/{tag}/processed/train/v7_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
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

                    print(f"Num objs {j}, saved batch {batch_idx}")

        # save the remaining data
        with h5py.File(f"datasets/{tag}/processed/train/v7_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
            hf.create_dataset(f'scenes', data=all_scenes_images)
            hf.create_dataset(f'success_grasps', data=all_success_grasps)
            hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
            hf.create_dataset(f'num_objs', data=all_num_objs)
        batch_idx += 1

        print(f"Num objs {j}, saved batch {batch_idx}")

        # save the evaluation data
        with h5py.File(f"datasets/{tag}/processed/eval/v7_bs_{batch_size}.h5", 'a') as hf:
            hf.create_dataset(f'scenes', data=eval_scenes_images)
            hf.create_dataset(f'success_grasps', data=eval_success_grasps)
            hf.create_dataset(f'failure_grasps', data=eval_failure_grasps)
            hf.create_dataset(f'num_objs', data=eval_num_objs)
        
        print(f'evaluation set has {len(eval_scenes_images)} samples')