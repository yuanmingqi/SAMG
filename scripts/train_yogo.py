import os
import time
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from torch.optim import lr_scheduler
from vit_pytorch import ViT
from loader import build_batch_loader, load_h5

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

class YOGO(nn.Module):
    def __init__(self,
                 scene_image_size=224,
                 grasp_dim=7,
                 scene_embed_dim=512, 
                 grasp_embed_dim=512,
                 ):
        super().__init__()

        self.scene_encoder = ViT(image_size = scene_image_size,
                                 patch_size = 32,
                                 num_classes = scene_embed_dim,
                                 dim = 1024,
                                 depth = 6,
                                 heads = 16,
                                 mlp_dim = 2048,
                                 dropout = 0.1,
                                 emb_dropout = 0.1
                                 )
        
        # # resnet encoder
        # self.scene_encoder = models.resnet18()
        # self.scene_encoder.fc = nn.Identity()

        self.grasp_encoder = nn.Sequential(
            nn.Linear(grasp_dim*3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, grasp_embed_dim)
        )
    
    def forward(self, scene_image, success_grasp, failure_grasp):
        scene_features = self.scene_encoder(scene_image)
        success_grasp = th.cat([success_grasp, th.sin(success_grasp), th.cos(success_grasp)], dim=1)
        failure_grasp = th.cat([failure_grasp, th.sin(failure_grasp), th.cos(failure_grasp)], dim=1)
        success_grasp_features = self.grasp_encoder(success_grasp)
        failure_grasp_features = self.grasp_encoder(failure_grasp)

        # scene_features = F.tanh(scene_features)
        # success_grasp_features = F.tanh(success_grasp_features)
        # failure_grasp_features = F.tanh(failure_grasp_features)

        return scene_features, success_grasp_features, failure_grasp_features
    
    def load(self, path, device):
        state_dict = th.load(path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)
        print(f"Loaded model from {path}")

    def triplet_loss(self, anchor, positive, negative, masks, margin):
        # # apply masks   
        # masks = masks.squeeze(0).unsqueeze(1)
        # # get reciprocal, transform masks into (0.5, 1)
        # def scale_tensor(tensor, x_min=1.0, x_max=10.0, y_min=1.0, y_max=0.5):
        #     scaled_tensor = y_min + (tensor - x_min) * (y_max - y_min) / (x_max - x_min)
        #     return scaled_tensor
        # masks = scale_tensor(masks)

        # performa normalization to features
        # anchor = F.normalize(anchor, p=2, dim=1)
        # positive = F.normalize(positive, p=2, dim=1)
        # negative = F.normalize(negative, p=2, dim=1)

        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        loss = F.relu(positive_distance - negative_distance + margin)
        # # apply masks
        # loss = loss * masks
        return loss.mean()
    
    def clip_loss(self, anchor, positive, negative, temperature=0.1):
        logits_per_image = th.matmul(anchor, positive.T) / temperature
        logits_per_text = logits_per_image.T

        # create  labels
        labels = th.arange(anchor.size(0), device=anchor.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
    
        loss = (loss_i2t + loss_t2i) / 2.0
        return loss
    
    def score_grasp_poses(self, scene_image, grasp_poses):
        self.eval()
        with th.no_grad():
            image_features = self.scene_encoder(scene_image)
            grasp_features = self.grasp_encoder(grasp_poses)
            scores = th.cosine_similarity(image_features.unsqueeze(0), grasp_features)
        return scores
    

def evaluate(model, eval_loader, device, margin):
    model.eval()
    correct = 0
    total = 0
    total_loss = th.tensor(0.0).to(device)
    with th.no_grad():
        for scene_images, success_grasps, failure_grasps, masks in eval_loader:
            scene_features, pos_features, neg_features = model(scene_images.squeeze(0).to(device), 
                                                              success_grasps.squeeze(0).to(device), 
                                                              failure_grasps.squeeze(0).to(device))
            eval_loss = model.module.triplet_loss(scene_features, pos_features, neg_features, masks, margin=margin)
            total_loss += eval_loss
            # get scores
            pos_scores = th.cosine_similarity(scene_features, pos_features)
            neg_scores = th.cosine_similarity(scene_features, neg_features)
            print(th.where(pos_scores > 0)[0].size())
            # get accuracy
            correct += (pos_scores > neg_scores).sum().item()
            total += pos_scores.size(0)
    
    accuracy = correct / total
    return accuracy, total_loss / len(eval_loader)


def main():
    # set model parameters
    scene_image_size = 224
    grasp_dim = 7
    scene_embed_dim = grasp_embed_dim = 128 
    margin = 1.0
    # set training parameters
    ## use multi-gpu
    device = th.device("cuda")
    num_epochs = 1000
    batch_size = 1024
    lr = 1e-3
    lr_decay_factor = 0.99
    lr_decay_step_size = 10
    weight_decay = 1e-4
    # load data    
    train_data = load_h5("./datasets/mixed/processed", 'train', batch_size)
    eval_data = load_h5("./datasets/mixed/processed", 'eval', batch_size)
    # build dataloaders
    train_loader, eval_loader = build_batch_loader(train_data, eval_data, batch_size, device)
    # build model
    yogo = YOGO(scene_image_size, grasp_dim, scene_embed_dim, grasp_embed_dim).to(device)
    # use multi-gpu
    yogo = nn.DataParallel(yogo)
    # load model
    # yogo.module.load("./logs/best/.pth", device)
    # set optimizer
    optimizer = th.optim.Adam(yogo.parameters(), lr=lr, weight_decay=weight_decay)
    # lr scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    # train loop
    for epoch in range(num_epochs):
        yogo.train()
        total_loss = th.tensor(0.0).to(device)
        t_s = time.perf_counter()
        for scene_images, success_grasps, failure_grasps, masks in train_loader:
            # forward pass
            optimizer.zero_grad()
            scene_features, pos_features, neg_features = yogo(scene_images.squeeze(0).to(device), 
                                                              success_grasps.squeeze(0).to(device), 
                                                              failure_grasps.squeeze(0).to(device))
            # compute loss
            loss = yogo.module.triplet_loss(scene_features, pos_features, neg_features, masks.to(device), margin=margin)
            # loss = yogo.module.clip_loss(scene_features, pos_features, neg_features)
            loss.backward()
            optimizer.step()
            total_loss += loss
        t_e = time.perf_counter()
        total_loss = total_loss / len(train_loader)
        accuracy, eval_loss = evaluate(yogo, eval_loader, device, margin)
        # update learning rate
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {accuracy:.4f}, Time: {t_e - t_s:.2f}s")

        # save model
        th.save(yogo.state_dict(), f"./logs/yogo_{epoch}_loss_{total_loss:.4f}_acc_{accuracy:.4f}.pth")


if __name__ == "__main__":
    main()
