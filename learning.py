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



if(len(sys.argv)<3):
    print("You have to specify the path as argument and the model ")
    exit()
path = sys.argv[1]
model = int(sys.argv[2])


#number of sift points
Siftvariable = 50
#number of dimensions to be reduced to with pca
pca_comon = 8


def pca_learning():
    All_Binary_FILES = os.listdir(path)
    List_descr = []
    for file_name in All_Binary_FILES:
        file_s = os.listdir(path + '/' + file_name)
        for file in file_s:
            if file.endswith('.png'):
                img = cv2.imread(path+ '/' + file_name + '/'
                                 + file)
                if np.shape(img)[0] < 64:
                    continue
                (keypoints, descriptors) = sift.detectAndCompute(img,
                        None)

                try:
                    for i in range(0, len(descriptors)):
                        List_descr.append(descriptors[i])
                except:
                    continue

                    # print ('file ', file)
    print("Number of elements in the input matrix before PCA = "+str(len(List_descr)*128))
    print("Start PCA ")
    result = pca.fit_transform(List_descr)
    print("Number of elements in the  input matrix  after PCA = ",str(len(List_descr)*pca_comon))
    print("Saving PCA .. Size of sift input vector reduced from "+str(128*Siftvariable)+" to "+str(pca_comon*Siftvariable))

    pk.dump(pca, open('pca.pkl', 'wb'))


def codification():
    All_Binary_FILES = os.listdir(path)
    codification_dict = {}
    i = 1
    for file_name in All_Binary_FILES:
        codification_dict[file_name] = i
        i = i + 1
    return codification_dict


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


def sift_using_pca(name):
    code = codification()
    All_Binary_FILES = os.listdir(path)
    dict_general = {}
    List_Features = []
    pca_1 = pk.load(open('pca.pkl', 'rb'))
    for file_name in All_Binary_FILES:
        file_s = os.listdir(path + '/' + file_name)
        filelist = [file for file in file_s if file.endswith('.png')]
        for file in file_s:
            List_descr = []

            if file.endswith('.png'):

                img = cv2.imread(path + '/' + file_name + '/'
                                 + file, 1)
                img_binary = cv2.imread(path + '/' + file_name
                        + '/' + file, 0)
                if np.shape(img)[0] < 64:
                    continue

                (keypoints, descriptors) = sift.detectAndCompute(img,
                        None)

                try:

                    List_descr = pca_1.transform(descriptors)
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
                        List_final = List_final[0:Siftvariable
                            * pca_comon]
                    List_final.append(aspect_ratio(img_binary.shape[0],
                            img_binary.shape[1]))

                    List_final.append(D(img_binary.shape[0],
                            img_binary.shape[1]))
                    (A, P) = compacity(img_binary)
                    List_final.append(A)
                    List_final.append(P)
                    List_final.append(code[file_name])
                    List_Features.append(List_final)
                except:
                    continue

                   # print file

    return List_Features


num_test_elemes = 400


def Learning_SVM(Features):
    #pour utilisé svm comme classifeur 

    if(model==0):
        print("Start the learning with SVM")
        Classification=svm.SVC()
    else:
        print("Start the learning with MLP")
        Classification = MLPClassifier(activation='logistic', solver='lbfgs'
                                   , warm_start=True, random_state=1,verbose=True,
                                   max_iter=20000)
                                   
    X = []
    Y = []

    random.shuffle(Features)

    result = []
    y_test = []
    predicted = []
    i = 700
    while i > 0:
        index = random.randrange(0, len(Features))
        result.append(Features.pop(index))
        i = i - 1

    for feature in Features:

        X.append(feature[0:len(feature) - 1])
        Y.append(feature[-1])
    y_test = []
    predicted = []

    for feature in result:

        y_test.append(feature[0:len(feature) - 1])
        predicted.append(feature[-1])

    Classification.fit(np.array(X), np.array(Y))

    cnt = 0
    for i in range(0, num_test_elemes):
        pred = Classification.predict([y_test[i]])
        if pred[0] == predicted[i]:
            cnt += 1

    print ('predictions: ' + str(cnt / num_test_elemes * 100) + '%')
    print('Saving model so it can be used in the tests files ')
    print('NOTE : in case of changing the pca_comon variable or Siftvariable  it has to be changed in the test units too before using it')

    pk.dump(Classification, open('SVM.pkl', 'wb'))






sift = cv2.xfeatures2d.SIFT_create(Siftvariable)
pca = PCA(n_components=pca_comon)



pca_learning()
Learning_SVM(sift_using_pca('pca.pkl'))



			
