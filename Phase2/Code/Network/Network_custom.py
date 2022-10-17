"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project

import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, gt):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    loss = nn.MSELoss()
    loss=loss(delta,gt)
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, patch_a,patch_b,gt):

        # print(batch.shape)
        # patch_a, patch_b, gt = batch
        stacked_img=torch.stack([patch_a,patch_b],dim=1)
        delta = self.model(stacked_img)
        loss = LossFn(delta, gt)
        loss = {"loss": loss}
        return loss

    def validation_step(self, patch_a,patch_b, gt):
        # patch_a, patch_b,gt = batch
        stacked_img=torch.stack([patch_a,patch_b],dim=1)
        delta = self.model(stacked_img)
        loss = LossFn(delta, gt)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, InputSize=10, OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,256,3)

        self.BN1=nn.BatchNorm2d(64)
        self.BN2=nn.BatchNorm2d(128)
        self.BN3=nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(7*7*256, 1024)
        self.fc2=nn.Linear(1024,OutputSize)

        self.pool = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.dropout=nn.Dropout(p=0.5)
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa,xb=None):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        # print(xa.shape)

        x = self.pool(self.BN1(F.relu(self.conv1(xa))))
        x = self.pool(self.BN1(F.relu(self.conv2(x))))

    
        x = self.pool(self.BN2(F.relu(self.conv3(x))))
        x = self.pool(self.BN3(F.relu(self.conv4(x))))


        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)


        return out
