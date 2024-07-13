import os
import time
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from vit_pytorch import ViT
from loader import build_eval_loader, load_h5

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
        self.scene_encoder = models.resnet18(pretrained=True)
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

    def triplet_loss(self, anchor, positive, negative, margin):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        loss = F.relu(positive_distance - negative_distance + margin)
        return loss.mean()
    
    def clip_loss(self, anchor, positive, negative, temperature):
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
    

def evaluate(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for scene_images, success_grasps, failure_grasps, masks in eval_loader:
            scene_features, pos_features, neg_features = model(scene_images.squeeze(0).to(device), 
                                                              success_grasps.squeeze(0).to(device), 
                                                              failure_grasps.squeeze(0).to(device))
            # get scores
            pos_scores = th.cosine_similarity(scene_features, pos_features)
            neg_scores = th.cosine_similarity(scene_features, neg_features)
            # print(pos_scores, neg_scores, th.where(pos_scores > 0), th.where(neg_scores > 0))
            # print masks of false predictions
            print(th.where(pos_scores > 0), th.mean(pos_scores - neg_scores))
            # get accuracy
            correct += (pos_scores > neg_scores).sum().item()
            total += pos_scores.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    # set model parameters
    scene_image_size = 224
    grasp_dim = 7
    scene_embed_dim = grasp_embed_dim = 128 
    # set training parameters
    ## use multi-gpu
    device = th.device("cuda")
    batch_size = 1024

    # load data    
    eval_data = load_h5("./datasets/mixed/processed", 'eval', batch_size)
    # build dataloaders
    eval_loader = build_eval_loader(eval_data, batch_size, device)
    # build model
    yogo = YOGO(scene_image_size, grasp_dim, scene_embed_dim, grasp_embed_dim).to(device)
    yogo = nn.DataParallel(yogo)
    # load model
    state_dict = th.load("logs/yogo_133_loss_0.3220_acc_0.8200.pth", map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    yogo.module.load_state_dict(new_state_dict)

    # train loop
    accuracy = evaluate(yogo, eval_loader, device)
    print(f"Eval Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
