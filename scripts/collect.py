import cv2
import time
import pickle
import numpy as np
import torch

import utils
from env.constants import WORKSPACE_LIMITS
from env.env_collect import Environment
from grasp_detetor import Graspnet

if __name__ == "__main__":
    num_episode = 10
    seed = 1234

    env = Environment(gui=True)
    env.seed(seed)

    # load graspnet
    graspnet = Graspnet()

    for episode in range(num_episode):
        env.reset()
        env.generate_lang_goal()
        # if episode < 1000:
        #     num_obj = 8
        #     reset = env.add_objects(num_obj, WORKSPACE_LIMITS)
        # else:
        #     num_obj = 15
        #     reset = env.add_objects(num_obj, WORKSPACE_LIMITS)

        env.add_fixed_objects()

        eps_rgb_images = []
        eps_grasp_poses = []
        eps_grasp_labels = []

        eps_step = 0
        vaild = True
        while True:
            color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            bbox_images, bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)

            # collect rgb images
            eps_rgb_images.append(color_image)

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
            with open(f"assets/episode_{episode}.pkl", "wb") as f:
                pickle.dump((eps_rgb_images, eps_grasp_poses, eps_grasp_labels), f)
            
            