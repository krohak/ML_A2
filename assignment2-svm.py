import numpy as np
from matplotlib import (pyplot as plt, patches as patches)
from PIL import Image 
from skimage.filters import threshold_local 

import os
from os import listdir
from os.path import isfile, join

from scipy.ndimage.measurements import center_of_mass
from sklearn.metrics import accuracy_score
import scipy.misc

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC



# area of bounding box
def rotate_bbox_area(img, deg):
    box = img.rotate(deg, expand=True).getbbox()
    return (box[3] - box[1]) * (box[2] - box[0])
    
def rotate_crop(img, deg, padding=0):
    img_rotate = img.rotate(deg, expand=True, resample=Image.BILINEAR)
    box = img_rotate.getbbox()
    if padding > 0:
        box = np.asarray(box) + [-padding, -padding, +padding, +padding]
    return img_rotate.crop(box)


tol_deg = 1
# smallest bounding box wihin -10~10 degrees rotation
def opt_rotate(img, padding=0):
    opt_deg = np.argmin(
        [rotate_bbox_area(img, i) for i in range(-tol_deg,tol_deg+1)]
        ) - tol_deg
    return rotate_crop(img, opt_deg, padding)

# downsampling
def img_reduce(img, side=28, mode=Image.ANTIALIAS):
    h = side + 1 
    w = int(side * img.width / img.height) + 1
    img_reduced = img.copy()
    # the reduced image size is (w-1, h-1)
    img_reduced.thumbnail((w, h), mode)
    return img_reduced


# convert PIL.Image object to numpy.Array, for training
def img2arr(img):
    return np.asarray(img.getdata(), dtype=np.uint8).reshape(img.height, img.width, -1)


# process single signature with transparent background
def process_one(img):
    return img_reduce(opt_rotate(img, padding=1).convert('LA'))


clf = SVC(C=4,gamma=0.01)


def train():
    os.system('sh arrange.sh')
    
    path_gen = 'original/genuine/'
    path_fog = 'original/forged/'
    positiveFiles = [ path_gen + f for f in listdir(path_gen) if isfile(join(path_gen, f))]
    negativeFiles = [ path_fog + f for f in listdir(path_fog) if isfile(join(path_fog, f))]
    
    data = []
    labels = []

    for file_type, files in enumerate([negativeFiles,positiveFiles]):
        for image_file in files:

            img = Image.open(image_file).convert('LA')
            img_reduced = process_one(img)
            img_arr = img2arr(img_reduced)[:,:,-1]

            # 28x28 box around the center of mass of signature
            center = np.round(center_of_mass(img_arr))
            h = img_arr.shape[0]
            left = int(center[1]) - h//2
            mat = img_arr[:, left:left+h]

            # binarize the image 
            mat = mat>127

            # convert from 28x28 to 784 
            mat = np.concatenate(mat)

            if (mat.shape == (784,)):
                data.append(mat)
                labels.append(file_type)
            else: 
                # print("Removing %s from dataset"%image_file)
                pass

    # print("Normalized %s signature arrays with dimension %s"%(len(data),data[1].shape))
    
    data_np = np.asarray(data)
    labels = np.asanyarray(labels)
    
    global clf
    # fit entire data
    clf.fit(data_np, labels) 
    
def test(image_file):
    global clf
    
    if image_file:
            
            try:
                img = Image.open(image_file).convert('LA')
            except Exception as e:
                print(e)
                return "Error in Filepath"
            
            img_reduced = process_one(img)
            img_arr = img2arr(img_reduced)[:,:,-1]

            # 28x28 box around the center of mass of signature
            center = np.round(center_of_mass(img_arr))
            h = img_arr.shape[0]
            left = int(center[1]) - h//2
            mat = img_arr[:, left:left+h]

            # binarize the image 
            mat = mat>127

            # convert from 28x28 to 784 
            mat = np.concatenate(mat)

            if (mat.shape == (784,)):
                # give mat to clf 
                mat = mat.reshape(1, -1)
                y_pred = clf.predict(mat)
                return bool(y_pred[0]) 
                
            else: 
                print("Non existent bounding box")
                return False
        
    
    else:
        return "No filepath specified"