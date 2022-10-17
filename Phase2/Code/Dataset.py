
import cv2
import glob
from random import randrange

import numpy as np

import torch
from torch.utils.data import Dataset

def readImage(path):

        img=cv2.imread(path)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        gray=cv2.resize(gray,(300,500))

        #patch will be 128*128 and perturbations can be -16 and +16 thus top left should go from 16 to 300-(16+128). Same for the y direction.

        return gray

class ImageDataset(Dataset):
    def __init__(self,img_dir):

        self.img_directory = img_dir
        # print(img_dir)
        # print(self.img_directory+"*")
        self.files=glob.glob(self.img_directory+"*")
        self.files.sort()


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        gray_img = readImage(img_path)

        active_patch_x=randrange(16,gray_img.shape[1]-(16+128))
        active_patch_y=randrange(16, gray_img.shape[0]-(16+128))

        patch_A=gray_img[active_patch_y:active_patch_y+128,active_patch_x:active_patch_x+128]
        normalized_patch_A=(patch_A-np.mean(patch_A))/255

        corners_A=[]
        corners_A.append([active_patch_y,active_patch_x])
        corners_A.append([active_patch_y+128,active_patch_x])
        corners_A.append([active_patch_y,active_patch_x+128])
        corners_A.append([active_patch_y+128,active_patch_x+128])

        corners_B=[]
        for li in corners_A:
            per_y=randrange(-16,16)
            per_x=randrange(-16,16)

            corners_B.append([li[0]+per_y,li[1]+per_x])

        transform=cv2.getPerspectiveTransform(np.float32(corners_A),np.float32(corners_B))
        transform=np.linalg.inv(transform)

        warped_img=cv2.warpPerspective(gray_img,transform,(gray_img.shape[1],gray_img.shape[0]))
        patch_B = warped_img[corners_A[0][0]:corners_A[1][0], corners_A[0][1]:corners_A[2][1]]

        normalized_patch_B=(patch_B-np.mean(patch_B))/255
        
        labels=[]
        for j in range(len(corners_A)):

            labels.append([corners_B[j][0]-corners_A[j][0],corners_B[j][1]-corners_A[j][1]])

        labels=np.reshape(labels,4*2)

        
        return torch.FloatTensor(normalized_patch_A),torch.FloatTensor(normalized_patch_B), torch.FloatTensor(labels)