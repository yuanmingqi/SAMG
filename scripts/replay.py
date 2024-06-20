import cv2
import time
import numpy as np
import torch

import utils
from env.constants import WORKSPACE_LIMITS
from env.env_collect import Environment
from grasp_detetor import Graspnet

if __name__ == "__main__":
    num_episode = 10000
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

        all_grasps = []

        while True:
            # color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            # bbox_images, bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)

            # # graspnet
            # pcd = utils.get_fuse_pointcloud(env)
            # # Note that the object poses here can be replaced by the bbox 3D positions with identity rotations
            # with torch.no_grad():
            #     object_poses = env.get_true_object_poses()
            #     grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, object_poses)

            # for idx, action in enumerate(grasp_pose_set):
            #     env.snapshot()
            #     success, done = env.step(action)
            #     print(f"Action: {idx}, {success == 1}")
            #     if success == 1:
            #         all_grasps.append(action)
            #         break
            #     else:
            #         env.restore()
            
            # if len(grasp_pose_set) == 0:
            #     np.save(f"assets/grasps_{episode}.npy", all_grasps)
            #     exit(0)
            #     break

            actions = np.load(f"assets/grasps_{episode}.npy")
            for idx, action in enumerate(actions):
                # env.snapshot()
                success, done = env.step(action)