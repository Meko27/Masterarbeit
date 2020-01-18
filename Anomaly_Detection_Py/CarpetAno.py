# Carpet images anomaly detection
import numpy as np
import cv2 
import os
import glob
import ntpath
from skimage.feature import hog
import skimage
import matplotlib.pyplot as plt
from AnoDetector import AnoDetector
from HOG_ground_dist import HOG_ground_dist


# Load images
path_normal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/train/good'
path_anomoal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/test/hole'
valid_img_type = '.png' # all images are type jpg
img_addrs_list_normal = glob.glob(path_normal + '/*' + valid_img_type)
img_addrs_list_anomal = glob.glob(path_anomoal + '/*' + valid_img_type)
img_addrs_list = img_addrs_list_normal + img_addrs_list_anomal
#img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_id_list = []
img_list = []
hog_list = []


# Params
# Image scale
img_scale = 0.2
# HOG:
orientations = 9
pixels_per_cell = (64, 64)
cells_per_block = (4, 4)
# Ano detector:
iter = 6
n_outliers = 17
k = 10

# Load images
for i,addr in enumerate(img_addrs_list): 
    img_id = ntpath.basename(addr)
    img_id = img_id[:-len(valid_img_type)]
    img_id_list.append(img_id) # list of img names
    img = cv2.imread(addr)
    img_size = img.shape[:2]
    #img = cv2.resize(img,tuple(int(img_scale * size) for size in img_size))
    img = cv2.resize(img,(512,512))
    img_list.append(img)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))

print('img loaded')
ground_dist,d_hog = HOG_ground_dist(img_list[0],cell_size=pixels_per_cell, block_size=cells_per_block,block_stride=(3,3))
print('ground_dist calculated, size:', ground_dist.shape)

# Extract Features
hog_mat = np.empty((len(img_addrs_list), int(d_hog))) 
for i,img in enumerate(img_list):
    hog_vect = hog(img,orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block)
    hog_list.append(hog_vect)
    #hog_mat[i] = hog_vect

hog_mat = np.array(hog_list)
print('HOG calculated, HOG size: ', hog_mat.shape)
print('calculated HOG length: ', d_hog)
# Calculate Anomalies
x = hog_mat
Ano_Detector = AnoDetector(x,k=k,iter = iter,metric='emd',grounddist=ground_dist)
print('AnoDetector constructed')
dist_vect_ano, idx_outliers,Ano_Detector = Ano_Detector.calc_outliers(n_outliers)
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', idx_outliers)



