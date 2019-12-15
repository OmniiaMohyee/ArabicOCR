# imports
import preprocessing/preprocess
import sementation/word_segment
import sementation/contour

# general intializations

# for data_point in data_points XD

    # read text file --> labels

    # read scanned file 
    # pre-process image
    clean_img = preproc(img_path)
    # segment image into characters
    words,size_words = word_seg(clean_img)
    # for each word:
    for word in words:
        connected_components = word_segment(word,threshold)
        # for each comp in connected_components:
        for comp in connected_components:
            segment(comp)
            # for each character:
                # extract feature vector 
                # add it to dataset.csv file
            # if #chars in text != in scanned --> ignore this word
