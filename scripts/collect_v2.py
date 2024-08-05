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
    parser.add_argument('--tag', type=str, default='single+')
    parser.add_argument('--num_episode', type=int, default=9000)
    parser.add_argument('--num_obj', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tag = args.tag
    num_episode = args.num_episode
    seed = args.seed
    num_obj = args.num_obj
    patch_size = 10

    env = Environment(gui=False)
    env.seed(seed)

    # load graspnet
    graspnet = Graspnet()

    num_avail_episodes = 0
    episode = 0
    while num_avail_episodes < num_episode:
        episode += 1
        env.reset()
        env.generate_lang_goal()
        # env.add_fixed_objects()
        _, _, obj_urdf_files, obj_init_poses, obj_init_orns = env.add_objects(num_obj, WORKSPACE_LIMITS)
        
        # get rgb and depth images  
        color_image, depth_image, mask_image = utils.get_true_heightmap(env)

        # graspnet
        pcd = utils.get_fuse_pointcloud(env)

        # Note that the object poses here can be replaced by the bbox 3D positions with identity rotations
        with torch.no_grad():
            object_poses = env.get_true_object_poses()
            grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, object_poses)
        
        # see if no available poses
        if len(grasp_pose_set) <= 1:
            continue
        else:
            num_grasp_poses = len(grasp_pose_set)

        # try all grasp poses
        success_indices = []
        failure_indices = []

        for idx, action in enumerate(grasp_pose_set):
            # take a snapshot
            env.snapshot()
            # execute the action
            success, done = env.step(action)

            if success == 1:
                label = 1
                success_indices.append(idx)
            else:
                label = 0
                failure_indices.append(idx)

            print(f"Episode {episode}, {num_grasp_poses} Poses, Action: {idx}, {success == 1}")

            if len(success_indices) >= patch_size and len(failure_indices) >= patch_size:
                break

            # restore the objects
            env.restore()

        samples = {
            'num_objs': num_obj,
            'rgb_image': color_image,
            'depth_image': depth_image, 
            'grasp_poses': grasp_pose_set, 
            'success_indices': success_indices, 
            'failure_indices': failure_indices
            }
        
        # if no successful or failed grasps, skip this episode
        if len(success_indices) == 0 or len(failure_indices) == 0:
            continue

        # save data
        stamp = episode ^ int.from_bytes(os.urandom(4), byteorder="little")
        with open(f"datasets/{tag}/trajs/episode_{stamp}_{num_obj}_objs.pkl", "wb") as f:
            pickle.dump({
                "obj_urdf_files": obj_urdf_files,
                "obj_init_poses": obj_init_poses,
                "obj_init_orns": obj_init_orns,
                "samples": samples
            }, f)

        num_avail_episodes += 1
        print(f"\n Available episodes={num_avail_episodes} \n")
            
            