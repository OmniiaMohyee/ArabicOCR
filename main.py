# imports
import preprocess
import word_segment
import contour
import statistical_features
#----------------
import glob
from PIL import Image
import cv2
from sklearn import preprocessing
import csv
import pandas as pd 
 
# general intializations
data_points = glob.glob("../clean_dataset/scanned/scanned/capr1.png")
labels = glob.glob('../clean_dataset/text/text/capr1.txt')
#prepare output file
csv_file = open('dataset.csv', encoding='utf-8', mode='w')
writer = csv.DictWriter(csv_file, fieldnames=['char'])
writer.writeheader()

text = []
words_iter = 0
word_features = []

for data_point in data_points:
    words_iter = 0
    word_features = []
    f = open(labels[0], encoding='utf-8')  # read text file --> labels
    for line in f:
        for word in line.split():
            text.append(word)     
    # read scanned file 
    clean_img = preprocess.preproc(data_point) # pre-process image
    words,size_words = word_segment.word_seg(clean_img) # segment image into characters
    
    print("size_words = ", len(words))
    for word in words:
        scanned_chars_count = 0
        # threshold = 1
        cv2.imwrite("word.png", word)
        # connected_components, _ = word_segment.word_segment(word,threshold)
        # print(connected_components)
        # print("len of connected_components = ",len(connected_components))
        # for comp in connected_components:
        #     print("comp", comp)
        #     cv2.imwrite("connected_comp.png", comp)
        chars = contour.segment(word)
        scanned_chars_count += len(chars)
        print("number of chars = ", len(chars))
        # for each character:
        for char in chars:
            cv2.imwrite("char.png", char)
            # extract feature vector 
            cropped_char = statistical_features.crop_image(char)
            feature_vector= statistical_features.getFeatureVector(cropped_char)
            word_features.append(feature_vector)
            print("feature_vector extracted ")
            # add it to dataset.csv file
        if len(text[words_iter]) != scanned_chars_count:
            print("number of segmented character is wrong :(")
            # print("len(text[words_iter])", len(text[words_iter]))
            #--> ignore this word but report it
            print(text[words_iter])
        else:
            #add to csv file
            for c in text[words_iter]:
                writer.writerow({'char': c})
            csv_file.close()
            # encode character labels into numbers
            df = pd.read_csv('dataset.csv')
            label_encoder = preprocessing.LabelEncoder() 
            df['code']= label_encoder.fit_transform(df['char']) # Encode labels in column 'char
            df.to_csv('dataset.csv')

            with open('dataset.csv', 'w') as csvoutput:
                with open('dataset.csv','r') as csvinput:
                    writerz = csv.writer(csvoutput, lineterminator='\n')
                    readerz = csv.reader(csvinput)

                    all = []
                    row = next(readerz)
                    for i in range (len(feature_vector)):
                        row.append(str('f')+i)
                    all.append(row) # for headers
                    k = 0 #char
                    
                    i = 0 #one feature for one char
                    for row in readerz:
                        for i in range (len(feature_vector)):
                            row.append(word_features[k][i])
                        all.append(row)
                        k += 1
                    writerz.writerows(all)
        words_iter += 1