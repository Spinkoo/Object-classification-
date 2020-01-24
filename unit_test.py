#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import os
from functools import reduce
from sklearn.decomposition import PCA
import numpy as np
import pickle as pk
import operator
import math
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neural_network import MLPClassifier
import random
import sys






if(len(sys.argv)<2):
    print("You have to specify the path as argument ")
    exit()
path = sys.argv[1]



def load_pca():
    pca_reload = pk.load(open('pca.pkl', 'rb'))
    return pca_reload


def aspect_ratio(x, y):
    return x / y


def D(x, y):
    return 4 * (x + y) ** 2 / (x * y)


def compacity(image):
    A = np.count_nonzero(image)
    count = cv2.findContours(image, cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)
    xc = cv2.drawContours(image, count[1], -1, (0, 255, 0), 3)
    P = np.count_nonzero(xc)

    return (A, P ** 2 / (4 * math.pi * A))




def load_clasification_image():
    pca_reload = pk.load(open('SVM.pkl', 'rb'))
    return pca_reload


def Feautre_images(img, img_binary):
    (keypoints, descriptors) = sift.detectAndCompute(img, None)

    # print img

    List_descr = pca_classification.transform(descriptors)
    List_final = []
    for item in List_descr:
        for item_x in item:
            List_final.append(item_x)

    i = len(List_final)
    while i < Siftvariable * pca_comon:
        for j in range(pca_comon):
            List_final.append(0)
        i += pca_comon
    if len(List_final) > Siftvariable * pca_comon:
        List_final = List_final[0:Siftvariable * pca_comon]

    List_final.append(aspect_ratio(img_binary.shape[0],
                      img_binary.shape[1]))
    List_final.append(D(img_binary.shape[0], img_binary.shape[1]))
    (A, P) = compacity(img_binary)
    List_final.append(A)
    List_final.append(P)
    return List_final


def classifier(features):
    return classification.predict([features])



codes=['bike','boat','canoe','car','human','noise','pickup','truck','van']
def codification(i):
    return codes[i-1]
def boundingRect_(image):
    image = cv2.blur(image, (9, 9))
    count = cv2.findContours(image, cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_NONE)[1]
    List_ = []
    List__ = []
    for cnt2 in count:
        if cv2.contourArea(cnt2) < 1000:
            continue
        (x2, y2, w2, h2) = cv2.boundingRect(cnt2)
        List_.append(image[y2:y2 + h2, x2:x2 + w2])
        L = [x2, y2, w2, h2]
        List__.append(L)

    return (List_, List__)


Siftvariable = 50
pca_comon = 8
sift = cv2.xfeatures2d.SIFT_create(Siftvariable)
pca = PCA(n_components=pca_comon)
pca_classification = load_pca()
classification = load_clasification_image()

def work():
    img=cv2.imread(path, 0)
    print(codification(classifier(Feautre_images(img,img))[0]))    
    
               

work()


			