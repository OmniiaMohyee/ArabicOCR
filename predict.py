# imports
import preprocess
import word_segment
import contour

import csv
import glob
import cv2
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
import time
from features.feat_test import crop_image,getFeatureVector

#Tasks
#1- load model
# load the model from disk
filename = 'savedmodels/trial1/knn88.11.sav'
decision_tree = pickle.load(open(filename, 'rb'))
# Y_pred = decision_tree.predict(X_test)
# result = loaded_model.score(X_test, Y_test)
# print(result)

#Load encoder
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('label_encoder.npy',allow_pickle=True)

#2- Read image from paths Tests
data_points = glob.glob("test/*.png")
for data_point in data_points:
    #3- start timer
    start_time= time.time()
    name = data_point[5:] # remove test folder from name
    name = name[:-4] # remove .png from name

    #4- Preprocess -> Segment -> get features (try not to write on disk)
    clean_img,clean_image_not_bin = preprocess.preproc(data_point) # pre-process image
    words = word_segment.word_seg(clean_img,clean_image_not_bin) # segment image into characters
    words_iter =0
    Text=[]
    ij = 0
    for word in words:
        chars = contour.segment(word, words_iter)
        words_iter += 1
        WordFeatures =[]
        #get features
        for i in range(len(chars)):
            ## Ask what i need to do here from these LINES !!!!!!
            # img_grey = cv2.cvtColor(chars[i],cv2.COLOR_RGB2GRAY)
            _, bw_img = cv2.threshold(chars[i],127,255,cv2.THRESH_BINARY) #convert to binary
            black_char = cv2.bitwise_not(bw_img) #back to black char
            cropped_img = crop_image(black_char)
            WordFeatures.append(getFeatureVector(cropped_img)) 

        #5- predict
        Y_pred = decision_tree.predict(WordFeatures)
        TransformedLabels = encoder.inverse_transform(Y_pred)
        Text.append(TransformedLabels)

    #6- calculate time
    end_time= time.time()
    taken_time= end_time - start_time
    print(taken_time)
    #7- write text
    with open('output/running_time.txt','a')as running:
        running.write(str(taken_time)+'\n')
    with open('output/text/'+name+'.txt','w',encoding='utf-8')as f:
        for i in range(len(Text)):
            for j in range(len(Text[i])):
                f.write(Text[i][j])
            f.write(' ')


#8- To get acc run edit.py to get edit distance
# python edit.py output/text realoutput

