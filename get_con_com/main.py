from preprocess import *
from word_segment import *
from trial import *

page_bin , page = preproc('tests/2.png')

all_words, all_words_bin = word_seg(page_bin,page)
print(len(all_words),len(all_words_bin))
for i in range(len(all_words)):
    #cv2.imwrite("word"+str(i+1)+".png",all_words_bin[i])
    get_components(all_words_bin[i],1,3,2,i+1)