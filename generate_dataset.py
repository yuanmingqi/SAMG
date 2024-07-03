import os
import glob
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sam_hq.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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

    # set sam model
    sam_checkpoint = "assets/sam_hq_vit_b.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # save all data
    all_scenes_images = []
    all_success_grasps = []
    all_failure_grasps = []

    # load pickle files
    # use tqdm to show progress bar
    for pkl_file in tqdm.tqdm(glob.glob(f"datasets/{tag}/*_{num_obj}_objs.pkl")):
        pkl_file_name = os.path.basename(pkl_file).split('.')[0]
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        rgb_image = data['samples']['rgb_image']
        depth_image = data['samples']['depth_image']

        # get masks and heightmaps
        ## generate masks
        masks = mask_generator.generate(rgb_image)
        seg_mask = np.zeros_like(masks[0]['segmentation'])
        center_points = []
        for mask in masks:
            if mask['area'] > int(0.25 * img_height * img_width) or mask['area'] < min_mask_area:
                continue
            seg_mask += mask['segmentation']
            center_points.append(np.array(mask['bbox']).astype(int))
        ## generate heightmap
        heightmap = generate_heatmap_image((img_height, img_width), center_points, sigma=heightmap_sigma)

        # save the mask and heightmap
        plt.imsave(f"datasets/{tag}/{pkl_file_name}_seg_mask.png", seg_mask)
        plt.imsave(f"datasets/{tag}/{pkl_file_name}_heightmap.png", heightmap)

        # get scene fusion images
        # print(seg_mask.shape, depth_image.shape, heightmap.shape)
        scene_image = np.stack([seg_mask, depth_image, heightmap])
        # print(scene_image.shape)

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
    np.savez(f"datasets/{tag}/processed_{num_obj}_objs.npz", 
             scenes=all_scenes_images, 
             success_grasps=all_success_grasps, 
             failure_grasps=all_failure_grasps)

        
        