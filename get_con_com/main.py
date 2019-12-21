from preprocess import *
from word_segment import *
from trial import *

page_bin , page = preproc('tests/2.png')

all_words, all_words_bin = word_seg(page_bin,page)
print(len(all_words),len(all_words_bin))
for i in range(len(all_words)):
    resized = resize_erode(all_words_bin[i],5,5,1)
    
    resized_non_bin =  resize_only(all_words[i],5)

    result_bin , result = draw_vert_hist(resized,resized_non_bin,2)
    cv2.imwrite("word"+str(i+1)+".png",result)
    #get_components(all_words_bin[i],1,3,2,i+1)