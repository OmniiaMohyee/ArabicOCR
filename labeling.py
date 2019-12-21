# imports
import preprocess
import word_segment
import contour
#---------------------
import csv
import glob
import cv2


def build_association_file():
    # general intializations
    data_points = glob.glob("../scanned_patch/*.png")
    labels = glob.glob('../text_patch/*.txt')
    tot_right = 0
    tot_wrong = 0
    data_point_it = 0
    text_chars_iter = 1  # for writing in csv file
    scanned_chars_iter = 0 # same
    #prepare output file
    csv_file = open('dataset.csv', encoding='utf-8', mode='w')
    writer = csv.DictWriter(csv_file, fieldnames=['char','path'])
    writer.writeheader()

    for data_point in data_points:
        right = 0
        wrong = 0 
        text = []
        words_iter = 0
        # read text file --> labels
        f = open(labels[data_point_it], encoding='utf-8')
        for line in f:
            for word in line.split():
                text.append(word) 
        print('actual number of words = ', len(text))    
        # read scanned file 
        clean_img,clean_image_not_bin = preprocess.preproc(data_point) # pre-process image
        words = word_segment.word_seg(clean_img,clean_image_not_bin) # segment image into characters
        print("num of segmented words = ", len(words))
        for word in words:
            scanned_chars_count = 0
            scanned_chars_iter = 0
            chars = contour.segment(word, words_iter)
            scanned_chars_count += len(chars)
            if len(text[words_iter]) != scanned_chars_count:
                wrong += 1
            else:
                right += 1
                #add to csv file
                for c in text[words_iter]:
                    cv2.imwrite("chars/char_"+str(text_chars_iter)+".png", chars[scanned_chars_iter])
                    writer.writerow({'char': c, 'path' : "chars/char_"+str(text_chars_iter)+".png"})
                    text_chars_iter += 1
                    scanned_chars_iter += 1                    
            words_iter += 1
        data_point_it += 1
        print("right = ", right)
        print("wrong = ", wrong)
        tot_right += right
        tot_wrong += wrong
    csv_file.close()
    print("total right = ", tot_right)
    print("total wrong = ", tot_wrong)
   