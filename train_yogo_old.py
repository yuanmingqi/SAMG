import os 
import time
import h5py
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import ViT

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class YOGO(nn.Module):
    def __init__(self,
                 scene_image_size=224,
                 grasp_dim=7,
                 scene_embed_dim=512, 
                 grasp_embed_dim=512,
                 ):
        super().__init__()

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

        self.grasp_encoder = nn.Sequential(
            nn.Linear(grasp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, grasp_embed_dim)
        )
    
    def forward(self, scene_image, success_grasp, failure_grasp):
        scene_features = self.scene_encoder(scene_image)
        success_grasp_features = self.grasp_encoder(success_grasp)
        failure_grasp_features = self.grasp_encoder(failure_grasp)
        return scene_features, success_grasp_features, failure_grasp_features

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        loss = F.relu(positive_distance - negative_distance + margin)
        return loss.mean()
    
    def score_grasp_poses(self, scene_image, grasp_poses):
        self.eval()
        with th.no_grad():
            image_features = self.scene_encoder(scene_image)
            grasp_features = self.grasp_encoder(grasp_poses)
            scores = th.cosine_similarity(image_features.unsqueeze(0), grasp_features)
        return scores
    
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
    
def build_loader(data, batch_size, device, num_workers=4):
    scenes, success_grasps, failure_grasps = data['scenes'], data['success_grasps'], data['failure_grasps']
    # split data into evaluation and training set
    num_data = len(data['scenes'])
    indices = np.random.permutation(num_data)
    split = int(0.8 * num_data)
    train_indices, eval_indices = indices[:split], indices[split:]
    # create dataloaders
    train_dataset = GraspDataset(scenes[train_indices], success_grasps[train_indices], failure_grasps[train_indices], device)
    eval_dataset = GraspDataset(scenes[eval_indices], success_grasps[eval_indices], failure_grasps[eval_indices], device)
    # build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
    return train_loader, eval_loader

def evaluate(model, eval_loader):
    model.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for scene_images, success_grasps, failure_grasps in eval_loader:
            scene_features, pos_features, neg_features = model(scene_images, success_grasps, failure_grasps)
            # get scores
            pos_scores = th.cosine_similarity(scene_features, pos_features)
            neg_scores = th.cosine_similarity(scene_features, neg_features)
            # get accuracy
            correct += (pos_scores > neg_scores).sum().item()
            total += pos_scores.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    # set model parameters
    scene_image_size = 224
    grasp_dim = 7
    scene_embed_dim = grasp_embed_dim = 512
    # set training parameters
    device = th.device("cuda")
    num_epochs = 1000
    batch_size = 512
    lr = 1e-3
    # load data
    # all_data = {}
    # file = 'datasets/mixed/processed_1_objs.h5'
    # with h5py.File(file, 'r') as f:
    #     all_data['scenes']= f['scenes'][:]
    #     all_data['success_grasps']= f['success_grasps'][:]
    #     all_data['failure_grasps']= f['failure_grasps'][:]
    file = 'datasets/processed_data.npz'
    all_data = np.load(file, allow_pickle=True)
    train_loader, eval_loader = build_loader(all_data, batch_size, device)
    # build model
    yogo = YOGO(scene_image_size, grasp_dim, scene_embed_dim, grasp_embed_dim).to(device)
    optimizer = th.optim.Adam(yogo.parameters(), lr=lr)

    # train loop
    yogo.train()
    for epoch in range(num_epochs):
        total_loss = 0.
        t_s = time.perf_counter()
        for scene_images, success_grasps, failure_grasps in train_loader:
            # forward pass
            optimizer.zero_grad()
            scene_features, pos_features, neg_features = yogo(scene_images, success_grasps, failure_grasps)
            loss = yogo.triplet_loss(scene_features, pos_features, neg_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        t_e = time.perf_counter()
        total_loss /= len(train_loader)
        accuracy = evaluate(yogo, eval_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Eval Acc: {accuracy:.4f}, Time: {t_e - t_s:.2f}s")


if __name__ == "__main__":
    main()