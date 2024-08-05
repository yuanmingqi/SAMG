import os
import glob
import tqdm
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_heatmap(center, shape, sigma):
    x = np.arange(0, shape[1], 1, float)  
    y = np.arange(0, shape[0], 1, float)[:, np.newaxis]  
    x0, y0 = center
    heatmap = np.exp(-4*np.log(2) * ((x - x0)**2 + (y - y0)**2) / sigma**2)
    return heatmap

def generate_heatmap_image(image_shape, boxes, sigma=10):
    heatmap = np.zeros(image_shape, dtype=np.float32)
    
    for box in boxes:
        x1, y1, w, h = box
        center = (x1 + w / 2, y1 + h / 2)
        heatmap += generate_gaussian_heatmap(center, image_shape, sigma)
    
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

if __name__ == '__main__':
    # environment settings
    tag = 'mixed'
    num_obj = 1
    img_width = 224
    img_height = 224
    min_mask_area = 100
    heightmap_sigma = 10

    # save all data
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []

    # load pickle files
    # use tqdm to show progress bar
    for pkl_file in tqdm.tqdm(glob.glob(f"datasets/{tag}/trajs/*_{num_obj}_objs.pkl")[:100]):
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

    # save the dataset
    # np.savez(f"datasets/{tag}/processed_{num_obj}_objs.npz", 
    #          scenes=all_scenes_images, 
    #          success_grasps=all_success_grasps, 
    #          failure_grasps=all_failure_grasps)

    with h5py.File(f"/media/HDD2/zihui/processed_1_objs.h5", 'a') as hf:
        hf.create_dataset(f'scenes', data=all_scenes_images)
        hf.create_dataset(f'success_grasps', data=all_success_grasps)
        hf.create_dataset(f'failure_grasps', data=all_failure_grasps)

        
        