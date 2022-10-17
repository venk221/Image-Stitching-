#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

from Dataset_test import ImageDataset
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network_custom import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch

import random

import torch.nn as nn


# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel()

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    OutSaveT = open(LabelsPathPred, "w")

    rand_idx=random.randint(0,len(TestSet)-1)

    
    test_loss=[]
    loss_fn=nn.MSELoss()

    for count in tqdm(range(len(TestSet))):

        if count==rand_idx:
            patch_A,patch_B,H4pt_gt,corners_A,corners_B,img_path=TestSet[count]
            img=cv2.imread(img_path)

            patch_A=patch_A.unsqueeze(0)
            patch_B=patch_B.unsqueeze(0)   

            stacked_img=torch.stack([patch_A,patch_B],dim=1)          

            H4pt_pred = model.model(stacked_img)
            # print(H4pt_pred.shape)
            
            loss=loss_fn(H4pt_pred,H4pt_gt.unsqueeze(0))

            test_loss.append(loss.item())


            H4pt_pred=H4pt_pred.tolist()[0]
            # print(H4pt_pred)



            corners_B_predicted=[]

            for k,corner in enumerate(corners_A):
                y,x=corner
                y=y+H4pt_pred[2*k]
                x=x+H4pt_pred[2*k+1]

                corners_B_predicted.append([y,x])



            corners_B_gt=np.array(corners_B)
            idx=[0,1,3,2]
            corners_B_gt=corners_B_gt[idx]
            corners_B_gt = corners_B_gt.reshape((-1, 1, 2))

            corners_B_predicted=np.array(corners_B_predicted)
            corners_B_predicted=corners_B_predicted[idx]
            corners_B_predicted = corners_B_predicted.reshape((-1, 1, 2))

            print(corners_B_gt)
            print(corners_B_predicted)



            image = cv2.polylines(img, [corners_B_gt],
                      True, [255,0,0],
                      1)
            cv2.imwrite('inter_img.png',image)
            image=cv2.imread('inter_img.png')

            image_2 = cv2.polylines(image, np.int32([corners_B_predicted]),
                      True, [0,0,255],
                      1)

            cv2.imwrite('Test_image_1.png',image_2)



            


        patch_A,patch_B,labels,corners_A,_,img_path = TestSet[count]

        patch_A=patch_A.unsqueeze(0)
        patch_B=patch_B.unsqueeze(0)

        stacked_img=torch.stack([patch_A,patch_B],dim=1)           

        # Img, ImgOrg = ReadImages(Img)
        H4pt = model.model(stacked_img)
        loss=loss_fn(H4pt,labels.unsqueeze(0))

        test_loss.append(loss.item())
        OutSaveT.write(str(H4pt) + "\n")

    OutSaveT.close()

    print(f' Test Mean Loss is: {np.mean(test_loss)}')


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/49model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../../P1TestSet/Phase2/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # test_dataset_dir='../../P1TestSet/Phase2/'

    test_dataset=ImageDataset(BasePath)

    # Setup all needed parameters including file reading
    ImageSize= None

    # Define PlaceHolder variables for Input and Predicted output
    LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    TestOperation(ImageSize, ModelPath, test_dataset, LabelsPathPred)

    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)


if __name__ == "__main__":
    main()
