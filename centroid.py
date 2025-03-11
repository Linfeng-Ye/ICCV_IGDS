import os
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

class Centroid():
    def __init__(self, n_classes, samples_data, samples_tar, Ctemp, CdecayFactor=0.9999):
        self.n_classes = n_classes
        self.centroids = torch.ones((n_classes, n_classes)) / n_classes
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.samples_data = torch.tensor(samples_data).to(self.device)
        self.samples_tar = samples_tar
    def update_epoch(self, model, data_loader):
        self.centroids = torch.zeros_like(self.centroids)
        model.train()
        device = next(model.parameters()).device
        for image,target in tqdm(data_loader):
            image,target = image.to(device), target.to(device)
            logit = model(image).detach()

            Classes =  target.cpu().unique()
            logit = logit.cpu()
            output = F.softmax(logit.float(), 1)
            '{}_'.format()
            for Class in Classes:
                self.centroids[Class] += torch.sum(output[target.cpu() == Class], axis = 0)

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])

    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target.cpu()).to(target.device)
    