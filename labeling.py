from sklearn import preprocessing
import csv
import pandas as pd 

# ======================== read text file =======================================
#prepare output file
csv_file = open('dataset.csv', encoding='utf-8', mode='w')
writer = csv.DictWriter(csv_file, fieldnames=['char'])
writer.writeheader()
#read text file and convert it into labels(chars)
f = open('capr2.txt', encoding='utf-8')
while True:
    c = f.read(1)
    if not c:
        break
    writer.writerow({'char': c})
csv_file.close()
# encode character labels into numbers
df = pd.read_csv('dataset.csv')
label_encoder = preprocessing.LabelEncoder() 
df['code']= label_encoder.fit_transform(df['char']) # Encode labels in column 'char
df.to_csv('dataset.csv')
# print(df) 

