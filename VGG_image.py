import random, csv
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

# cluster center file
root = "/YOUR/CLUSTER/FILE/DIRECTORY/"
cluster_file = os.path.join(root, "clustering_train_labels.csv")


def load_labeled_data(img_path,train = True):
    gt_path = img_path.replace('.png','.h5').replace('images','ground_truth')

    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    results = {}
    with open(os.path.join(root, cluster_file), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            results[row[0]] = int(row[1])

    file_name = img_path.split('.png')[0].split('/')[-1]
    cluster_center = results[file_name]

    return img, target, cluster_center


def load_unlabeled_data(img_path,train = True):
    img = Image.open(img_path).convert('RGB')
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img, img_gray


def load_unlabeled_data2(img_path,train = True):
    img = Image.open(img_path).convert('RGB')
    img2 = Image.open(img_path).convert('RGB')
    
    #crop the image region
    height, width = img[:,:,0].shape
    img_reg = img[0:200, 0:200,:]
    img2_reg = img2[16:, 16:, :]

    return img_reg, img2_reg


def load_val_data(img_path,train = True):

    gt_path = img_path.replace('.png','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    
    return img, target