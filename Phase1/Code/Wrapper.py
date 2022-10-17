# Add any python libraries here

import enum
import numpy as np
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle

import copy

from scipy.ndimage import maximum_filter

import random

def minmax(image):
  image=(image - np.min(image))/np.ptp(image)
  return image

def getHarris(image,photo_index=1):


  # print('Hello')

	original_img=copy.deepcopy(image)

	operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray_image = np.float32(operatedImage)
	prob_scores = cv2.cornerHarris(gray_image, 2, 3,0.04)#0.15 is free parameter
	prob_scores = cv2.dilate(prob_scores, None)

	
	mask=prob_scores>0.005*prob_scores.max()

	image[mask]=[0, 0, 255]

	fig=plt.figure()
	plt.imshow(image)
	plt.savefig(f'Corners_{photo_index}.png')

	# print(mask.sum())



	#   cv2.imshow('Corners',image)
	#   cv2.waitKey(0)

	return original_img,prob_scores

def doANMS(original_img,prob_scores,best_points=1000,photo_index=1):


  # n_strong=mask.sum()

	temp_img=copy.deepcopy(original_img)

	local_maximas = maximum_filter(prob_scores,size=30)
	new_mask=(prob_scores==local_maximas)
	temp_mask=(prob_scores>0)
	new_mask=np.logical_and(new_mask,temp_mask)

	indexes=np.argwhere(new_mask==True)

	# print(indexes)

	n_strong=indexes.shape[0]
	r=float('inf')*np.ones((n_strong,))

	ED=0


	for i in tqdm(range(n_strong),total=n_strong):
		for j in range(n_strong):

			x_i=indexes[i][0]
			y_i=indexes[i][1]
			x_j=indexes[j][0]
			y_j=indexes[j][1]

			if prob_scores[x_i,y_i]>prob_scores[x_j,y_j]:

				ED=(x_i-x_j)**2+(y_i-y_j)**2

			if r[i]>ED:
				r[i]=ED

	# print(r)


	# with open('./r_1.pkl','rb') as f:
	# 	r=pickle.load(f)

	#Sort descending 
	top_indexes=(-r).argsort()[:best_points]
	top_indexes=indexes[top_indexes]

	print("Number of final points after ANMS:",top_indexes.shape)

	for x,y in top_indexes:

		x_int=y
		y_int=x

		cv2.circle(temp_img,(x_int,y_int),2,(0,0,255),-1)



	fig=plt.figure()
	plt.imshow(temp_img)
	plt.savefig(f'ANMS_{photo_index}.png')

	# with open('r_1.pkl','wb') as f:
	# 	pickle.dump(r,f)

	top_points=top_indexes

	return top_points


def getFeatures(original_img,top_indexes):

	gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

	gray_img=np.pad(gray_img,(20,20),mode='edge')


	features=[]

	for x,y in top_indexes:

		x_int=y
		y_int=x


		small_patch=gray_img[x:x+40,y:y+40]

		gauss_patch=cv2.GaussianBlur(small_patch,(3,3),cv2.BORDER_DEFAULT)

		subsampled=gauss_patch[0::5,0::5]
		feature=np.reshape(subsampled,(64,1))

		feature=((feature-np.mean(feature))/(np.std(feature)))

		features.append(feature)
	

	features=np.array(features)
	# print(features.shape)


	return features


def featureMapping(features_1,features_2,threshold=0.95):

	print('Feature Mapping ...')

	matched_pairs=[]

	for i,feature_1 in enumerate(features_1):

		lowest_dist=float('inf')
		second_lowest_dist=float('inf')

		best_index=0

		for j, feature_2 in enumerate(features_2):

			distance=(feature_1-feature_2)**2
			distance=np.sum(distance)
			distance=np.sqrt(distance)

			if distance < lowest_dist:
				second_lowest_dist=lowest_dist
				lowest_dist=distance

				best_index=j

			elif distance < second_lowest_dist:
				second_lowest_dist=distance

			
			

		
		ratio=lowest_dist/second_lowest_dist

		if ratio<threshold:
			matched_pairs.append([i,best_index])

	print(len(matched_pairs))


	return matched_pairs

def visualize_pairs(image_1,image_2,top_points_1,top_points_2,matched_pairs,title='Mapping'):

	temp_img1=copy.deepcopy(image_1)
	temp_img2=copy.deepcopy(image_2)

	keypoints1=[]
	keypoints2=[]

	match_vector=[]

	### Converting data into required data to visualize matches. Into KeyPoints and DMatch datatypes.

	for x,y in top_points_1:
		keypoints1.append(cv2.KeyPoint(int(y),int(x),3))
	
	for x,y in top_points_2:
		keypoints2.append(cv2.KeyPoint(int(y),int(x),3))

	for li in matched_pairs:
		match_vector.append(cv2.DMatch(li[0],li[1],1))

	matchedImg=np.array([])

	matchedImg = cv2.drawMatches(temp_img1, keypoints1, temp_img2, keypoints2, match_vector,matchedImg)

	cv2.imwrite(f'{title}.jpg',matchedImg)

	return matchedImg



def doRANSAC(matched_pairs,top_points_1,top_points_2,iterations=1000,distance_threshold=500):

	print('Doing RANSAC')

	# count_epochs=0
	max_inliers=45
	good_match_flag=False
	inliers1=[]
	inliers2=[]
	matched_pairs_final=[]

	homography_matrix=np.array([])

	H_best=None

	if len(matched_pairs)<4:
		return matched_pairs,homography_matrix,good_match_flag

	for count_epochs in tqdm(range(iterations),total=iterations):
		temp_inlier_1=[]
		temp_inlier_2=[]

		temp_matching_pairs=[]

		random_indexes=random.sample(range(len(matched_pairs)),4)

		pts_1=[]
		pts_2=[]

		for i in range(4):
			pts_1.append(top_points_1[matched_pairs[random_indexes[i]][0]])
			pts_2.append(top_points_2[matched_pairs[random_indexes[i]][1]])

		pts_1=np.array(pts_1).astype('float32')
		pts_2=np.array(pts_2).astype('float32')

		H=cv2.getPerspectiveTransform(pts_1,pts_2)


		for k in range(len(matched_pairs)):

			point_1=top_points_1[matched_pairs[k][0]]
			point_2=top_points_2[matched_pairs[k][1]]

			point_1_projected = np.append(point_1, [1], axis = 0)
			point_1_projected = np.matrix.transpose(point_1_projected)
			
			point_1_projected = np.matmul(H,point_1_projected)

			point_1_projected=np.delete(point_1_projected,2)

			dist=(point_1_projected-point_2)**2
			dist=np.sum(dist)

			if dist < distance_threshold:
				temp_matching_pairs.append(matched_pairs[k])
				temp_inlier_1.append(top_points_1[matched_pairs[k][0]])
				temp_inlier_2.append(top_points_2[matched_pairs[k][1]])

		
	
			

		if len(temp_matching_pairs)>max_inliers:
			H_best=H
			matched_pairs_final=temp_matching_pairs
			inliers1=temp_inlier_1
			inliers2=temp_inlier_2

			break


	if len(matched_pairs_final)>4:
		good_match_flag=True


	print("Final matching pairs are:",len(matched_pairs_final))

	inliers1=np.array(inliers1)
	inliers2=np.array(inliers2)


	homography_matrix,_ = cv2.findHomography(np.float32(inliers1),np.float32(inliers2))


	

	
	return matched_pairs_final,homography_matrix,good_match_flag





def doAll():

	photo_index=1

	base1 = cv2.imread("../Data/Train/Set1/1.jpg")
	original_img_1,prob_scores=getHarris(base1,photo_index)
	top_points_1=doANMS(original_img_1,prob_scores,photo_index=photo_index)
	features_1 = getFeatures(original_img_1,top_points_1)

	photo_index+=1


	base2=cv2.imread("../Data/Train/Set1/2.jpg")
	original_img_2,prob_scores=getHarris(base2,photo_index)
	top_points_2=doANMS(original_img_2,prob_scores,photo_index=photo_index)
	features_2 = getFeatures(original_img_2,top_points_2)

	matched_pairs=featureMapping(features_1,features_2)

	matchedImg=visualize_pairs(original_img_1,original_img_2,top_points_1,top_points_2,matched_pairs,title='beforeRansac')

	final_matched_pairs,homography_matrix,flag=doRANSAC(matched_pairs,top_points_1,top_points_2)

	matchedImg_ransac=visualize_pairs(original_img_1,original_img_2,top_points_1,top_points_2,final_matched_pairs,title='afterRansac')

	


# if __name__=='main':
#   main()

doAll()
