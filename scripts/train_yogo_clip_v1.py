import os
import time
import h5py
import tqdm
import glob
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import ViT
from torch.optim import lr_scheduler
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

class YOGO(nn.Module):
    def __init__(self,
                 scene_image_size=224,
                 grasp_dim=7,
                 scene_embed_dim=512, 
                 grasp_embed_dim=512,
                 temperature=0.07
                 ):
        super().__init__()
        self.temperature = temperature
        # self.scene_encoder = ViT(image_size = scene_image_size,
        #                          patch_size = 32,
        #                          num_classes = scene_embed_dim,
        #                          dim = 1024,
        #                          depth = 6,
        #                          heads = 16,
        #                          mlp_dim = 2048,
        #                          dropout = 0.1,
        #                          emb_dropout = 0.1
        #                          )

        # resnet encoder
        self.scene_encoder = models.resnet18()
        self.scene_encoder.fc = nn.Identity()
        # success grasp encoder
        self.grasp_encoder = nn.Sequential(
            nn.Linear(grasp_dim*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, grasp_embed_dim)
        )

    def forward(self, scene_image, success_grasp, failure_grasp):
        scene_features = self.scene_encoder(scene_image)
        success_grasp = th.cat([success_grasp, th.sin(success_grasp), th.cos(success_grasp)], dim=1)
        failure_grasp = th.cat([failure_grasp, th.sin(failure_grasp), th.cos(failure_grasp)], dim=1)
        success_grasp_features = self.grasp_encoder(success_grasp)
        failure_grasp_features = self.grasp_encoder(failure_grasp)
        return scene_features, success_grasp_features, failure_grasp_features

    def sym_clip_loss(self, scene_features, sg_features, fg_features):
        sg_scene = self.clip_loss(scene_features, sg_features)
        fg_scene = self.clip_loss(scene_features, fg_features)
        return sg_scene - fg_scene + 15
    
    def clip_loss(self, image_embeds, text_embeds):
        """
        image_embeds: Tensor of shape (batch_size, embed_dim)
        text_embeds: Tensor of shape (batch_size, embed_dim)
        """
        batch_size, embed_dim = image_embeds.shape

        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)
        
        # Compute similarity matrix
        logits_per_image = th.matmul(image_embeds, text_embeds.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        # Labels are just the indices of the diagonal
        labels = th.arange(batch_size, device=image_embeds.device)
        
        # Compute cross entropy loss
        loss_img_to_txt = F.cross_entropy(logits_per_image, labels)
        loss_txt_to_img = F.cross_entropy(logits_per_text, labels)
        
        # Total loss is the average of the two losses
        total_loss = (loss_img_to_txt + loss_txt_to_img) / 2
        return total_loss
    
    def load(self, path, device):
        state_dict = th.load(path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)
        print(f"Loaded model from {path}")
    
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
        scene_images = th.as_tensor(scene_images).float()
        success_grasps = th.as_tensor(success_grasps).float()
        failure_grasps = th.as_tensor(failure_grasps).float()
        return scene_images, success_grasps, failure_grasps
    
def build_loader(data, batch_size, device, num_workers=4):
    scenes, success_grasps, failure_grasps = data['scenes'], data['success_grasps'], data['failure_grasps']

    # split data into evaluation and training set
    num_data = len(data['scenes'])
    indices = np.random.permutation(num_data)
    split = int(0.9 * num_data)
    train_indices, eval_indices = indices[:split], indices[split:]
    # create dataloaders
    train_dataset = GraspDataset(scenes[train_indices], success_grasps[train_indices], failure_grasps[train_indices], device)
    eval_dataset = GraspDataset(scenes[eval_indices], success_grasps[eval_indices], failure_grasps[eval_indices], device)
    # build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)#, num_workers=num_workers)
    return train_loader, eval_loader

def load_h5_file(file):
    data = {
        'scenes': [],
        'success_grasps': [],
        'failure_grasps': [],
        # 'num_objs': []
    }
    with h5py.File(file, 'r') as f:
        data['scenes'] = f['scenes'][:]
        data['success_grasps'] = f['success_grasps'][:]
        data['failure_grasps'] = f['failure_grasps'][:]
        # data['num_objs'].append(f['num_objs'][:])
    return data

def evaluate(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for scene_images, success_grasps, failure_grasps in eval_loader:
            scene_features, pos_features, neg_features = model(scene_images.to(device), 
                                                              success_grasps.to(device), 
                                                              failure_grasps.to(device))
            # get scores
            pos_scores = th.cosine_similarity(scene_features, pos_features)
            neg_scores = th.cosine_similarity(scene_features, neg_features)
            # get accuracy
            correct += (pos_scores > neg_scores).sum().item()
            # print false predictions
            # print(th.where(pos_scores <= neg_scores), th.where(pos_scores > 0)[0].size())
            total += pos_scores.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    # set seed
    seed = 1
    th.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.cuda.manual_seed_all(seed)
    # set model parameters
    scene_image_size = 224
    grasp_dim = 28
    scene_embed_dim = 512 
    grasp_embed_dim = 512
    temperature = 0.1
    # set training parameters
    device = th.device("cuda")
    num_epochs = 5000
    batch_size = 512
    lr = 1e-4
    lr_decay_factor = 0.99
    lr_decay_step_size = 10
    weight_decay = 1e-4
    # load data    
    data = load_h5_file("datasets/single/clip_dataset.h5")
    # build dataloaders
    train_loader, eval_loader = build_loader(data, batch_size, device)
    # build model
    yogo = YOGO(scene_image_size, grasp_dim, scene_embed_dim, grasp_embed_dim, temperature).to(device)
    # create parallel model
    yogo = nn.DataParallel(yogo)
    # load model
    yogo.module.load("logs/yogo_clip_0.9020_v1.pth", device)
    optimizer = th.optim.Adam(yogo.parameters(), lr=lr, weight_decay=weight_decay)
    # lr scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    bench = 0.5

    # train loop
    for epoch in range(num_epochs):
        total_loss = th.tensor(0.0).to(device)
        t_s = time.perf_counter()
        for scene_images, success_grasps, failure_grasps in train_loader:
            # forward pass
            optimizer.zero_grad()
            scene_features, pos_features, neg_features = yogo(scene_images.to(device), 
                                                              success_grasps.to(device), 
                                                              failure_grasps.to(device))
            loss = yogo.module.sym_clip_loss(scene_features, pos_features, neg_features)
            loss.backward()
            optimizer.step()
            
            total_loss += loss
        t_e = time.perf_counter()
        total_loss /= len(train_loader)
        accuracy = evaluate(yogo, eval_loader, device)
        # update lr
        # scheduler.step()
        print(f"E {epoch+1}/{num_epochs}, L: {total_loss:.4f}, Acc: {accuracy:.4f}, T: {t_e - t_s:.2f}s")

        if accuracy > bench:
            # try to remove old model
            if os.path.exists(f"./logs/yogo_clip_{bench:.4f}_v1.pth"):
                os.remove(f"./logs/yogo_clip_{bench:.4f}_v1.pth")
            bench = accuracy
            th.save(yogo.state_dict(), f"./logs/yogo_clip_{accuracy:.4f}_v1.pth")
            print(f"\nE {epoch+1}/{num_epochs}, Acc: {accuracy:.4f}\n")

if __name__ == "__main__":
    main()
