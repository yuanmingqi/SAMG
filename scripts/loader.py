import glob, tqdm, h5py
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

class GraspDataset(Dataset):
    def __init__(self, scenes, success_poses, failure_poses, device):
        self.scenes = scenes
        self.success_poses = success_poses
        self.failure_poses = failure_poses
        self.device = device

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_images = self.scenes[idx]
        success_grasps = self.success_poses[idx]
        failure_grasps = self.failure_poses[idx]
        scene_images = th.as_tensor(scene_images, device=self.device).float()
        success_grasps = th.as_tensor(success_grasps, device=self.device).float()
        failure_grasps = th.as_tensor(failure_grasps, device=self.device).float()
        return scene_images, success_grasps, failure_grasps
    
class GraspBatchDataset(Dataset):
    def __init__(self, scenes, success_poses, failure_poses, masks, device):
        self.scenes = scenes
        self.success_poses = success_poses
        self.failure_poses = failure_poses
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_images = self.scenes[idx]
        success_grasps = self.success_poses[idx]
        failure_grasps = self.failure_poses[idx]
        masks = self.masks[idx]
        # print(idx, scene_images.shape, success_grasps.shape, failure_grasps.shape)
        scene_images = th.as_tensor(scene_images).float()
        success_grasps = th.as_tensor(success_grasps).float()
        failure_grasps = th.as_tensor(failure_grasps).float()
        masks = th.as_tensor(masks).float()
        
        return scene_images, success_grasps, failure_grasps, masks
    
# def build_loader(data, batch_size, device, num_workers=4):
#     scenes, success_grasps, failure_grasps = data['scenes'], data['success_grasps'], data['failure_grasps']
#     # split data into evaluation and training set
#     num_data = len(data['scenes'])
#     indices = np.random.permutation(num_data)
#     split = int(0.8 * num_data)
#     train_indices, eval_indices = indices[:split], indices[split:]
#     # create dataloaders
#     train_dataset = GraspDataset(scenes[train_indices], success_grasps[train_indices], failure_grasps[train_indices], device)
#     eval_dataset = GraspDataset(scenes[eval_indices], success_grasps[eval_indices], failure_grasps[eval_indices], device)
#     # build dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=num_workers)
#     eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
#     return train_loader, eval_loader

def build_batch_loader(train_data, eval_data, batch_size, device, num_workers=4):
    # create datasets
    train_dataset = GraspBatchDataset(train_data['scenes'], 
                                      train_data['success_grasps'], 
                                      train_data['failure_grasps'], 
                                      train_data['num_objs'], 
                                      device)
    eval_dataset = GraspBatchDataset(eval_data['scenes'], 
                                     eval_data['success_grasps'], 
                                     eval_data['failure_grasps'], 
                                     eval_data['num_objs'], 
                                     device)
    # build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_loader, eval_loader

def build_eval_loader(eval_data, batch_size, device, num_workers=4):
    # create datasets
    eval_dataset = GraspBatchDataset(eval_data['scenes'], 
                                     eval_data['success_grasps'], 
                                     eval_data['failure_grasps'], 
                                     eval_data['num_objs'], 
                                     device)
    # build dataloaders
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    return eval_loader

def load_h5(path, flag, batch_size):
    all_data = {
        'scenes': [],
        'success_grasps': [],
        'failure_grasps': [],
        'num_objs': []
    }
    file_list = glob.glob(f"{path}/{flag}/v7_bs_{batch_size}*.h5")#[:1]
    # load all h5 files
    for file in tqdm.tqdm(file_list):
        with h5py.File(file, 'r') as f:
            all_data['scenes'].append(f['scenes'][:])
            all_data['success_grasps'].append(f['success_grasps'][:])
            all_data['failure_grasps'].append(f['failure_grasps'][:])
            all_data['num_objs'].append(f['num_objs'][:])
    print(f"{flag} data loaded from {len(file_list)} files.")

    return all_data