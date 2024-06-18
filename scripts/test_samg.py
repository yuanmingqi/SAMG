import argparse
import numpy as np
import random
import torch
import os

import utils
from env.constants import WORKSPACE_LIMITS
from env.environment_samg import Environment
from logger import Logger
from grasp_detetor import Graspnet
from models.samg_sac import SAMG
from sam_hq.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.models import resnet18
import torch.nn as nn

def sam_resnet_fusion(sam, resnet, color_img, depth_img, device):
    with torch.no_grad():
        depth_img = torch.as_tensor(depth_img, dtype=torch.float32).unsqueeze(0).to(device)
        
        sam.set_image(color_img)

        # center_points = []
        # num_grids = 40
        # x = np.linspace(0, 224, num_grids).astype(int)
        # y = np.linspace(0, 224, num_grids).astype(int)

        # center_points = np.array([[i, j] for i in x for j in y])

        # for i in range(num_grids):
        #     for j in range(num_grids):
        #         center_points.append([interval + interval * i, interval + interval * j])
        # center_points = np.array(center_points)

        segs = []
        for idx, num_grids in enumerate([16, 32, 64]):
            center_points = []

            interval = 224 // num_grids

            for i in range(num_grids):
                for j in range(num_grids):
                    center_points.append([i*interval+interval//2, j*interval+interval//2])
            center_points = np.array(center_points)

            mask, _, _ = sam.predict(
                    point_coords=center_points,
                    point_labels=[1 for _ in range(len(center_points))],
                    multimask_output=False,
                )
            segs.append(mask)

        # masks = sam.generate(color_img)
        # segs = []
        # for mask in masks:
        #     # too large mask will be ignored
        #     if mask['area'] > int(0.25 * 224 * 224):
        #         continue
        #     else:
        #         segs.append(mask['segmentation'])
        segs = torch.as_tensor(segs, dtype=torch.float32).to(device)
        # print(segs.shape)
        features = resnet(segs.permute(1, 0, 2, 3))
        # print(features.shape)
        # quit(0)
        # segs_hmap = torch.cat((segs, depth_img), dim=0)
        # downsample the segmentation masks and height map from 1*224*224 to 1*512
        # features = resnet(segs_hmap.unsqueeze(1).repeat(1, 3, 1, 1))

    return features

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_false', default=True)
    parser.add_argument('--testing_case_dir', action='store', type=str, default='testing_cases/')
    parser.add_argument('--testing_case', action='store', type=str, default=None)

    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')

    parser.add_argument('--num_episode', action='store', type=int, default=15)
    parser.add_argument('--max_episode_step', type=int, default=8)

    # Transformer paras
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    
    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=True)
    env.seed(args.seed)
    # load logger
    logger = Logger(case_dir=args.testing_case_dir)
    # load graspnet
    graspnet = Graspnet()
    # load sam model
    sam_checkpoint = "assets/sam_hq_vit_b.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    # sam_mask_generator = SamAutomaticMaskGenerator(sam)
    sam_mask_generator = SamPredictor(sam)
    # load resnet
    resnet = resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.to(device=args.device)
    # build agent
    agent = SAMG(grasp_dim=7, args=args)
    logger.load_checkpoint(agent, 
                            #    args.model_path,
                               '/home/birl/code/SAMG/logs/2024-06-18-14-31-58-train/checkpoints/sac_checkpoint_2024-06-18_15-42-35_71.pth', 
                               args.evaluate)
        
    if os.path.exists(args.testing_case_dir):
        filelist = os.listdir(args.testing_case_dir)
        filelist.sort(key=lambda x:int(x[4:6]))
    if args.testing_case != None:
        filelist = [args.testing_case]
    case = 0
    iteration = 0
    for f in filelist:
        f = os.path.join(args.testing_case_dir, f)

        logger.episode_reward_logs = []
        logger.episode_step_logs = []
        logger.episode_success_logs = []
        for episode in range(num_episode):
            episode_reward = 0
            episode_steps = 0
            done = False
            reset = False

            while not reset:
                env.reset()
                lang_goal = env.generate_lang_goal()
                reset = env.add_objects(12, WORKSPACE_LIMITS)
                # reset, lang_goal = env.add_object_push_from_file(f)
                # print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

            while not done:
                # check if one of the target objects is in the workspace:
                # out_of_workspace = []
                # for obj_id in env.target_obj_ids:
                #     pos, _, _ = env.obj_info(obj_id)
                #     if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                #         or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                #         out_of_workspace.append(obj_id)
                # if len(out_of_workspace) == len(env.target_obj_ids):
                #     print("\033[031m Target objects are not in the scene!\033[0m")
                #     break     


                color_image, depth_image, mask_image = utils.get_true_heightmap(env)

                bbox_images, bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)
                # graspnet
                pcd = utils.get_fuse_pointcloud(env)
                # Note that the object poses here can be replaced by the bbox 3D positions with identity rotations
                with torch.no_grad():
                    grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses())
                print("Number of grasping poses", len(grasp_pose_set))
                if len(grasp_pose_set) == 0:
                    break
                # preprocess
                # remain_bbox_images, bboxes, grasps = utils.preprocess(bbox_images, grasp_pose_set, args.patch_size)
                remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size))
                logger.save_bbox_images(iteration, remain_bbox_images)
                logger.save_heightmaps(iteration, color_image, depth_image)
                if bboxes == None:
                    break
                
                sam_features = sam_resnet_fusion(sam_mask_generator, resnet, color_image, depth_image, args.device)
                if len(grasp_pose_set) == 1:
                    action_idx = 0
                else:
                    with torch.no_grad():
                        # logits, action_idx, clip_probs, vig_attn = agent.select_action(bboxes, pos_bboxes, lang_goal, grasps, evaluate=args.evaluate)
                        logits, action_idx = agent.select_action(sam_features, grasps)

                action = grasp_pose_set[action_idx]
                reward, done = env.step(action)
                iteration += 1
                episode_steps += 1
                episode_reward += reward
                print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))

                if episode_steps == 15:
                    break

            
            logger.episode_reward_logs.append(episode_reward)
            logger.episode_step_logs.append(episode_steps)
            logger.episode_success_logs.append(done)
            logger.write_to_log('episode_reward', logger.episode_reward_logs)
            logger.write_to_log('episode_step', logger.episode_step_logs)
            logger.write_to_log('episode_success', logger.episode_success_logs)
            print("\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, episode_steps, round(episode_reward, 2), done))

            if episode == num_episode - 1:
                avg_success = sum(logger.episode_success_logs)/len(logger.episode_success_logs)
                avg_reward = sum(logger.episode_reward_logs)/len(logger.episode_reward_logs)
                avg_step = sum(logger.episode_step_logs)/len(logger.episode_step_logs)
                
                success_steps = []
                for i in range(len(logger.episode_success_logs)):
                    if logger.episode_success_logs[i]:
                        success_steps.append(logger.episode_step_logs[i])
                if len(success_steps) > 0:
                    avg_success_step = sum(success_steps) / len(success_steps)
                else:
                    avg_success_step = 1000

                result_file = os.path.join(logger.result_directory, "case" + str(case) + ".txt")
                with open(result_file, "w") as out_file:
                    out_file.write(
                        "%s %.18e %.18e %.18e %.18e\n"
                        % (
                            avg_success,
                            avg_step,
                            avg_success_step,
                            avg_reward,
                        )
                    )
                case += 1
                print("\033[034m average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(avg_step, avg_success_step, avg_reward, avg_success))