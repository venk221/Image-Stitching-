"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

from unittest.mock import patch
import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project

import pytorch_lightning as pl

import kornia.geometry.transform as geometry

# Don't generate pyc codes
sys.dont_write_bytecode = True


class TensorDLT(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,corners_A,h4pt_predicted):
        
        # print(corners_A.size())

        corners_A=torch.permute(corners_A,(1,0))
        batch_size,_=corners_A.shape

        # H4pt_pred=h4pt_predicted.tolist()[0]

        h4pt_predicted=torch.reshape(h4pt_predicted,(batch_size,4,2))

        # print(h4pt_predicted.shape)

        # corners_B=[]
        # for k,(y,x) in enumerate(corners_A):

        

        #     y_b=y+H4pt_pred[2*k]
        #     x_b=x+H4pt_pred[2*k+1]

        #     corners_B.append([y_b,x_b])


        # corners_A_list=corners_A[0]+corners_A[1]+corners_A[2]+corners_A[3]
        # corners_B_list=corners_B[0]+corners_B[1]+corners_B[2]+corners_B[3]

        # # corners_A=torch.stack(corners_A_list)
        # corners_B=torch.stack(corners_B_list)

        # corners_A=torch.permute(corners_A,(1,0))
        # corners_B=torch.permute(corners_B,(1,0))

        # batch_size,_=corners_A.shape

        corners_A=torch.reshape(corners_A,(batch_size,4,2))
        # corners_B=torch.reshape(corners_B,(batch_size,4,2))

        ##This code does very similar work to collating and making the batch to according to what we want####

        H_final=[]


        for batch_no in range(batch_size):

            A = []
            b= []

            for i in range(4):

                y_a,x_a=corners_A[batch_no][i]
                diff_y,diff_x=h4pt_predicted[batch_no][i]

                y_a,x_a=y_a.item(),x_a.item()

                y_b=y_a+diff_y
                x_b=x_a+diff_x

                # print(y_a)

                A_first_row=[0,0,0,-y_a,-x_a,-1,x_b*y_a,x_b*x_a]
                A_second_row=[y_a,x_a,1,0,0,0,-y_b*y_a,-y_b*x_a]


                A.append(A_first_row)
                A.append(A_second_row)

                b.append(-x_b)
                b.append(y_b)

                # break


            A=np.array(A,dtype=np.float)
            b=np.array(b,dtype=np.float)

            # print(A)
            # print(b)


            H = np.linalg.solve(A, b)

            H=H.flatten()

            H_new=[]

            for k,hi in enumerate(H):

                H_new.append(hi)

            
            H_new.append(1)
            H_new=np.reshape(H_new,(3,3))

            H_final.append(H_new)

        
        H_final=torch.FloatTensor(np.array(H_final))


        return H_final

class STN(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,patch_A,homography_matrix):
        patch_A=patch_A.unsqueeze(1)
        out = geometry.warp_perspective(patch_A, homography_matrix, (128, 128), align_corners=True)

        return out






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

def Photometric_LossFn(patch_B,predicted_patch_B):

    loss=nn.L1Loss()
    patch_B=patch_B.squeeze(1)
    patch_B.requires_grad_()
    loss=loss(patch_B,predicted_patch_B)

    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, patch_a,patch_b,gt,corner_a):

        # print(batch.shape)
        # patch_a, patch_b, gt = batch
        stacked_img=torch.stack([patch_a,patch_b],dim=1)
        patch_B_predicted,h4pt_predicted = self.model(stacked_img,patch_a,corner_a)
        loss = Photometric_LossFn(patch_B_predicted, patch_b)
        loss = {"loss": loss,"h4pt":h4pt_predicted}
        return loss

    def validation_step(self, patch_a,patch_b, gt,corner_a):
        # patch_a, patch_b,gt = batch
        stacked_img=torch.stack([patch_a,patch_b],dim=1)
        patch_B_predicted,h4pt_predicted  = self.model(stacked_img,patch_a,corner_a)
        loss = Photometric_LossFn(patch_B_predicted,patch_b)
        return {"val_loss": loss,"h4pt":h4pt_predicted}

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

        self.DLT=TensorDLT()
        self.st_network=STN()

        self.pool = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.dropout=nn.Dropout(p=0.5)

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


            


    def forward(self, xa,patch_A,corners_A,xb=None):
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
        # print(x.shape)
        x = self.pool(self.BN1(F.relu(self.conv2(x))))

        # print(x.shape)
        x = self.pool(self.BN2(F.relu(self.conv3(x))))
        # print(x.shape)
        x = self.pool(self.BN3(F.relu(self.conv4(x))))

        # print(x.shape)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        h4pt_predicted = self.fc2(x)

        # corners_A=torch.FloatTensor(corners_A).requires_grad_()

        homography_matrix=self.DLT(corners_A,h4pt_predicted)
        homography_matrix.requires_grad_()
        homography_matrix=homography_matrix.to(self.device)

        patch_A.requires_grad_()


        

        patch_B_predicted=self.st_network(patch_A,homography_matrix)



        


        return patch_B_predicted,h4pt_predicted
