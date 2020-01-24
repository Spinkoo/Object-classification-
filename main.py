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




class MG:

    def __init__(
        self,
        numOfGauss=1,
        height=100,
        width=100,
        ):

        self.width = width
        self.height = height
        self.numOfGauss = numOfGauss
        self.mus = np.zeros((height, width), np.double)
        self.sigmas = np.zeros((height, width), np.double)
        self.alpha = 0.1
        self.first = True
        self.beta=0.1
        self.mean=0.0
        self.diff = np.zeros((height, width), np.double)


    def calculate(self, frame):

        if self.first:
            self.prevFrame = self.mus = frame
            for i in range(self.height):
                for j in range(0, self.width):
                    self.sigmas[i][j]=127
            self.first = False
            return
        self.diff=cv2.absdiff(frame,self.mus)
        self.diff/=(self.sigmas)**0.5
        sigmas=self.alpha*abs(self.mus - frame)**2+(1-self.alpha)*self.sigmas
        mus=self.alpha*frame+(1-self.alpha)*self.mus
        
        for i in range(self.height):
                for j in range(0, self.width):
                    if(self.diff[i][j]<5):
                        self.mus[i][j]=mus[i][j]
                        self.sigmas[i][j]=sigmas[i][j]

        self.prevFrame = frame

    def substract(self, frame):
        diff = np.zeros((self.height, self.width), np.uint8)
        for i in range(self.height):
                for j in range(0, self.width):
                    if(self.diff[i][j]>3.5):
                        diff[i][j]=255
        return diff


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


    return List_Features


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
    (h, w) = np.shape(cv2.imread('b/1.png', 0))
    subtractor = MG(height=h, width=w)
    
    
    
    i=1
    kernel = np.ones((5,5),np.uint8)
    
    images = os.listdir('b')
    for file_name in images:
        img = cv2.imread('b' + '/' + str(i)+".png", 0)
        img1 = cv2.imread('b' + '/' + str(i)+".png", 1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
        img = img.astype(float)
        subtractor.beta=subtractor.alpha=1/i
        subtractor.calculate(img)
        i+=1
        if(i<10):
            continue
        fgMask = subtractor.substract(img)
        (List_, List__) = boundingRect_(fgMask)
        for x in range(len(List__)):
            cv2.rectangle(img1, (List__[x][0], List__[x][1]), (List__[x][0]
                          + List__[x][2], List__[x][1] + List__[x][3]),
                          (255, 0, 0), 2)
    
            try:
                ret, thresh1 = cv2.threshold(List_[x], 5, 255,
                        cv2.THRESH_BINARY)
                thresh1 = cv2.dilate(thresh1,kernel,iterations = 1)
    
    
                cv2.putText(img1, str(codification(classifier(Feautre_images(thresh1,
                            thresh1))[0])), (List__[x][0] + 8, List__[x][1]+15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0))
                cv2.imshow('', img1)
                cv2.waitKey(200)
            except Exception as e:
                print(e)

work()


			