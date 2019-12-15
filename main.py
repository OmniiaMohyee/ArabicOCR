# imports
import preprocess
import word_segment
import contour
import statistical_features
#----------------
import glob
from PIL import Image
import cv2
 
# general intializations
data_points = glob.glob("../clean_dataset/scanned/scanned/*.png")
labels = glob.glob('../clean_dataset/text/text/*.txt')
# print("data_points len = ", len(data_points))
# print("labels len = ", len(labels)) 

for data_point in data_points:
    # read text file --> labels

    # read scanned file 
    # pre-process image
    clean_img = preprocess.preproc(data_point)
    # segment image into characters
    words,size_words = word_segment.word_seg(clean_img)
    # for each word:
    for word in words:
        threshold = 1
        cv2.imwrite("word.png", word)
        connected_components, _ = word_segment.word_segment(word,threshold)
        print(connected_components)
        print("len of connected_components = ",len(connected_components))
        # for each comp in connected_components:
        for comp in connected_components:
            print("comp", comp)
            cv2.imwrite("connected_comp.png", comp)
            chars = contour.segment(comp)
            # for each character:
            for char in chars:
                # extract feature vector 
                cropped_char = statistical_features.crop_image(char)
                FeatureVector= statistical_features.getFeatureVector(cropped_char)
                print("FeatureVector >> ", FeatureVector)
                # add it to dataset.csv file
            # if #chars in text != in scanned --> ignore this word
