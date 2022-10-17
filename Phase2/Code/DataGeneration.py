#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

from asyncio import PidfdChildWatcher
import numpy as np
import cv2
from random import randrange

import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm
import argparse

# Add any python libraries here

def readImage(path):

    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray=cv2.resize(gray,(300,500))

    #patch will be 128*128 and perturbations can be -16 and +16 thus top left should go from 16 to 300-(16+128). Same for the y direction.

    return gray



def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataSplit', default='Train', help='Data split to make data -- can be Train, Test or Val')
    Parser.add_argument('--NumExamples',default=5000,type=int,help="Number of examples to process")

    Args = Parser.parse_args()
    data_split = Args.DataSplit
    num_examples=Args.NumExamples

    """
    Read image and generate datasets
    """

    dataset_file_name=f'{data_split}_dataset.pkl'
    dataset_dict={}
    dataset_size=num_examples

    print(f'Creating data for {dataset_file_name}')

    for i in tqdm(range(dataset_size),total=dataset_size):

        path=f'../Data/{data_split}/{i+1}.jpg'
        gray_img=readImage(path)

        # print(gray_img.shape)

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


        dataset_dict[i+1]={'patch_A':normalized_patch_A,'patch_B':normalized_patch_B,'label':labels}









        

        # if i==10:
        #     break


    with open(dataset_file_name,'wb') as f:
        pickle.dump(dataset_dict,f)





    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
