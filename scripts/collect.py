import os
import cv2
import time
import pickle
import numpy as np
import torch
import argparse

import utils
from env.constants import WORKSPACE_LIMITS
from env.env_collect import Environment
from grasp_detetor import Graspnet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='high', help='high, mid, low')
    parser.add_argument('--num_episode', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tag = args.tag
    num_episode = args.num_episode
    seed = args.seed

    env = Environment(gui=True)
    env.seed(seed)

    # load graspnet
    graspnet = Graspnet()

    for episode in range(num_episode):
        env.reset()
        env.generate_lang_goal()
        if tag == 'high':
            num_obj = 15
        elif tag == 'mid':
            num_obj = 8
        elif tag == 'low':
            num_obj = 3
        else:
            raise ValueError("Invalid tag")

        # env.add_fixed_objects()
        _, _, obj_urdf_files, obj_init_poses, obj_init_orns = env.add_objects(num_obj, WORKSPACE_LIMITS)

        eps_rgb_images = []
        eps_depth_images = []
        eps_grasp_poses = []
        eps_grasp_labels = []

        eps_step = 0
        vaild = True
        while True:
            color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            bbox_images, bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)

            # collect rgb and depth images
            eps_rgb_images.append(color_image)
            eps_depth_images.append(depth_image)

            # graspnet
            pcd = utils.get_fuse_pointcloud(env)
            # Note that the object poses here can be replaced by the bbox 3D positions with identity rotations
            with torch.no_grad():
                object_poses = env.get_true_object_poses()
                grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, object_poses)
                num_grasp_poses = len(grasp_pose_set)

            if len(grasp_pose_set) == 0:
                break
            else:
                num_grasp_poses = len(grasp_pose_set)
                eps_grasp_poses.append(np.array(grasp_pose_set))
                labels = np.ones(num_grasp_poses) * 7

            for _ in range(num_grasp_poses):
                idx = np.random.choice(num_grasp_poses, size=1, replace=False)[0]
                env.snapshot()
                action = grasp_pose_set[idx]
                success, done = env.step(action)
                print(f"Episode {episode}, Step: {eps_step}, Action: {idx}, {success == 1}")
                if success == 1:
                    labels[idx] = 1
                    break
                else:
                    labels[idx] = 0
                    env.restore()

            # collect grasp labels
            eps_grasp_labels.append(labels)

            eps_step += 1

            if eps_step >= 100:
                vaild = False
                break
        
        if vaild:
            # save data
            stamp = episode ^ int.from_bytes(os.urandom(4), byteorder="little")
            with open(f"datasets/{tag}/episode_{stamp}.pkl", "wb") as f:
                pickle.dump({
                    "obj_urdf_files": obj_urdf_files,
                    "obj_init_poses": obj_init_poses,
                    "obj_init_orns": obj_init_orns,
                    "rgb_images": eps_rgb_images,
                    "depth_images": eps_depth_images,
                    "grasp_poses": eps_grasp_poses,
                    "grasp_labels": eps_grasp_labels
                }, f)
            
            