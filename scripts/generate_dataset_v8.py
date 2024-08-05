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
    tag = 'single'
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []
    all_num_objs = []
    current_size = 0
    batch_idx = 0
    batch_size = 1024
    eval_size = 64   

    eval_scenes_images = []
    eval_success_grasps = []
    eval_failure_grasps = []
    eval_num_objs = []

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
            all_num_objs.append(num_objs)
            current_size += 1

            if current_size == (batch_size+eval_size):
                all_scenes_images = np.array(all_scenes_images)
                all_success_grasps = np.array(all_success_grasps)
                all_failure_grasps = np.array(all_failure_grasps)
                all_num_objs = np.array(all_num_objs)

                # split into train and eval
                indices = np.random.permutation(all_scenes_images.shape[0])
                train_indices, eval_indices = indices[:-eval_size], indices[-eval_size:]

                # save the batch
                with h5py.File(f"datasets/{tag}/final/train_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
                    hf.create_dataset(f'scenes', data=all_scenes_images[train_indices])
                    hf.create_dataset(f'success_grasps', data=all_success_grasps[train_indices])
                    hf.create_dataset(f'failure_grasps', data=all_failure_grasps[train_indices])
                    hf.create_dataset(f'num_objs', data=all_num_objs[train_indices])

                eval_scenes_images.extend(all_scenes_images[eval_indices].tolist())
                eval_success_grasps.extend(all_success_grasps[eval_indices].tolist())
                eval_failure_grasps.extend(all_failure_grasps[eval_indices].tolist())
                eval_num_objs.extend(all_num_objs[eval_indices].tolist())

                # reset the data
                all_scenes_images = []
                all_success_grasps = []
                all_failure_grasps = []
                all_num_objs = []
                current_size = 0
                batch_idx += 1

                print(f"saved train batch {batch_idx}")

        for i in range(len(data['samples']['failure_indices'])):
            success = random.choice(data['samples']['success_indices'])
            # randomly select a failure grasp
            failure = data['samples']['failure_indices'][i]
            success_grasp = data['samples']['grasp_poses'][success]
            failure_grasp = data['samples']['grasp_poses'][failure]

            # create pairs
            all_scenes_images.append(scene_image)
            all_success_grasps.append(success_grasp)
            all_failure_grasps.append(failure_grasp)
            all_num_objs.append(num_objs)
            current_size += 1

            if current_size == (batch_size+eval_size):
                all_scenes_images = np.array(all_scenes_images)
                all_success_grasps = np.array(all_success_grasps)
                all_failure_grasps = np.array(all_failure_grasps)
                all_num_objs = np.array(all_num_objs)

                # split into train and eval
                indices = np.random.permutation(all_scenes_images.shape[0])
                train_indices, eval_indices = indices[:-eval_size], indices[-eval_size:]

                # save the batch
                with h5py.File(f"datasets/{tag}/final/train_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
                    hf.create_dataset(f'scenes', data=all_scenes_images[train_indices])
                    hf.create_dataset(f'success_grasps', data=all_success_grasps[train_indices])
                    hf.create_dataset(f'failure_grasps', data=all_failure_grasps[train_indices])
                    hf.create_dataset(f'num_objs', data=all_num_objs[train_indices])

                eval_scenes_images.extend(all_scenes_images[eval_indices].tolist())
                eval_success_grasps.extend(all_success_grasps[eval_indices].tolist())
                eval_failure_grasps.extend(all_failure_grasps[eval_indices].tolist())
                eval_num_objs.extend(all_num_objs[eval_indices].tolist())

                # reset the data
                all_scenes_images = []
                all_success_grasps = []
                all_failure_grasps = []
                all_num_objs = []
                current_size = 0
                batch_idx += 1

                print(f"saved train batch {batch_idx}")

    # save the remaining data
    with h5py.File(f"datasets/{tag}/final/train_bs_{batch_size}_no_{batch_idx}.h5", 'a') as hf:
        hf.create_dataset(f'scenes', data=all_scenes_images)
        hf.create_dataset(f'success_grasps', data=all_success_grasps)
        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)
        hf.create_dataset(f'num_objs', data=all_num_objs)
    batch_idx += 1

    print(f"saved remaining train batch {batch_idx}")

    eval_scenes_images = np.array(eval_scenes_images)
    eval_success_grasps = np.array(eval_success_grasps)
    eval_failure_grasps = np.array(eval_failure_grasps)
    eval_num_objs = np.array(eval_num_objs)

    num_eval_samples = len(eval_scenes_images)
    # split into batches of batch_size
    for i in range(num_eval_samples//batch_size + 1):
        if i == num_eval_samples//batch_size:
            # last batch
            eval_scenes_images_batch = eval_scenes_images[i*batch_size:]
            eval_success_grasps_batch = eval_success_grasps[i*batch_size:]
            eval_failure_grasps_batch = eval_failure_grasps[i*batch_size:]
            eval_num_objs_batch = eval_num_objs[i*batch_size:]
        else:
            eval_scenes_images_batch = eval_scenes_images[i*batch_size:(i+1)*batch_size]
            eval_success_grasps_batch = eval_success_grasps[i*batch_size:(i+1)*batch_size]
            eval_failure_grasps_batch = eval_failure_grasps[i*batch_size:(i+1)*batch_size]
            eval_num_objs_batch = eval_num_objs[i*batch_size:(i+1)*batch_size]

        # save the batch
        with h5py.File(f"datasets/{tag}/final/eval_bs_{batch_size}_no_{i}.h5", 'a') as hf:
            hf.create_dataset(f'scenes', data=eval_scenes_images_batch)
            hf.create_dataset(f'success_grasps', data=eval_success_grasps_batch)
            hf.create_dataset(f'failure_grasps', data=eval_failure_grasps_batch)
            hf.create_dataset(f'num_objs', data=eval_num_objs_batch)

    # # save the evaluation data
    # with h5py.File(f"datasets/{tag}/final/eval_bs_{batch_size}.h5", 'a') as hf:
    #     hf.create_dataset(f'scenes', data=eval_scenes_images)
    #     hf.create_dataset(f'success_grasps', data=eval_success_grasps)
    #     hf.create_dataset(f'failure_grasps', data=eval_failure_grasps)
    #     hf.create_dataset(f'num_objs', data=eval_num_objs)
        
    print(f'evaluation set has {num_eval_samples} samples')