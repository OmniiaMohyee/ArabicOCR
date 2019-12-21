# imports
import preprocess
import word_segment
import contour
import statistical_features
import feat_test
#---------------------
import csv
import pandas as pd 
import glob
import cv2
from sklearn import preprocessing
from PIL import Image

def build_association_file():
    # general intializations
    data_points = glob.glob("../Pattern Recognition/clean_dataset/scanned/scanned/capr2.png")
    labels = glob.glob('../Pattern Recognition/clean_dataset/text/capr2.txt')
    right = 0
    wrong = 0 
    text = []
    text_chars_iter = 1
    scanned_chars_iter = 0
    #prepare output file
    csv_file = open('dataset.csv', encoding='utf-8', mode='w')
    writer = csv.DictWriter(csv_file, fieldnames=['char','code','path'])
    writer.writeheader()

    for data_point in data_points:
        words_iter = 0
        f = open(labels[0], encoding='utf-8')  # read text file --> labels
        for line in f:
            for word in line.split():
                text.append(word) 
        # print('actual number of words = ', len(text))    
        # read scanned file 
        clean_img,clean_image_not_bin = preprocess.preproc(data_point) # pre-process image
        words = word_segment.word_seg(clean_img,clean_image_not_bin) # segment image into characters
        chars = contour.segment(words[41], words_iter)

        # print("num of segmented words = ", len(words))
    #     for word in words:
    #         scanned_chars_count = 0
    #         scanned_chars_iter = 0
    #         # threshold = 1
    #         # cv2.imwrite("t/word"+str(words_iter)+".png", word)
    #         # connected_components, _ = word_segment.word_segment(word,threshold)
    #         # print("len of connected_components = ",len(connected_components))
    #         chars = contour.segment(word, words_iter)
    #         scanned_chars_count += len(chars)
    #         #print("number of chars = ", len(chars))
    #         if len(text[words_iter]) != scanned_chars_count:
    #             wrong += 1
    #             # print("number of segmented character is wrong :(")
    #             # print("len(text[words_iter])", len(text[words_iter]))
    #             #--> ignore this word but report it
    #             #print(text[words_iter])
    #         else:
    #             # print("number of segmented character is right Yaaaaa :)")
    #             right += 1
    #             #add to csv file
    #             for c in text[words_iter]:
    #                 cv2.imwrite("chars/char_"+str(text_chars_iter)+".png", chars[scanned_chars_iter])
    #                 writer.writerow({'char': c, 'path' : "chars/char_"+str(text_chars_iter)+".png"})
    #                 text_chars_iter += 1
    #                 scanned_chars_iter += 1
    #         words_iter += 1
    #         if words_iter >= len(text): #just to prevent craching till word segmentation is right
    #             break
    # print("rights = ", right)
    # print("wrong = ", wrong)
    csv_file.close()
    # encode character labels into numbers
    # df = pd.read_csv('dataset.csv')
    # label_encoder = preprocessing.LabelEncoder() 
    # df['code']= label_encoder.fit_transform(df['char']) # Encode labels in column 'char
    # df.to_csv('dataset.csv')