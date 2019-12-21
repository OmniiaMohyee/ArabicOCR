import pandas as pd
import numpy as np
import cv2 
import pickle
from features.feat_test import crop_image,getFeatureVector

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('dataset.csv')
Paths = train_df["path"]
Labels = train_df["char"]

Paths = list(Paths)
Labels = list(Labels)
print(Labels)
# print(Labels)
# print(Paths)
Features =[]
i=0
for path in Paths:
    # print(path)
    img = cv2.imread(path)
    # print(img.shape)
    img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, bw_img = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY) #convert to binary
    black_char = cv2.bitwise_not(bw_img) #back to black char
    cropped_img = crop_image(black_char)
    i+=1
    print(i)
    cv2.imwrite('train/'+str(i)+'.png',cropped_img)
    # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #black char
    # image_gray = cv2.bitwise_not(img_gray) #white char
    # image_thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    # cropped_img = crop_image(image_thresholded)
    Features.append(getFeatureVector(cropped_img)) 
    # print(getFeatureVector(cropped_img))
print(len(Features))
print(len(Labels))

#tasks
#1- read image from path
#2- read labels and encode them
#3- map each image to the feature vector
#4- divide the dataset into training and test set -----> lesssaaaaaa
#5- write the output of predict into a file ----->>>
#6- save the model 

logreg = LogisticRegression()
logreg.fit(Features, Labels)
# Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(Features, Labels) * 100, 2)
print(acc_log)
# save the model to disk
filename = 'Logistic_regression.sav'
pickle.dump(model, open(filename, 'wb'))

# # Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(acc_log)

# # Support Vector Machines
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print(acc_svc)


# #KNN
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print(acc_knn)


# # Gaussian Naive Bayes
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print(acc_gaussian)

# # Perceptron
# perceptron = Perceptron(tol= not None)
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print(acc_perceptron)

# # Linear SVC
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(acc_linear_svc)

# # Stochastic Gradient Descent
# sgd = SGDClassifier(tol= not None)
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print(acc_sgd)

# # Decision Tree
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)

# # Random Forest
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, Y_train)
# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# print(acc_random_forest)

# models = pd.DataFrame({
#     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#               'Random Forest', 'Naive Bayes', 'Perceptron', 
#               'Stochastic Gradient Decent', 'Linear SVC', 
#               'Decision Tree'],
#     'Score': [acc_svc, acc_knn, acc_log, 
#               acc_random_forest, acc_gaussian, acc_perceptron, 
#               acc_sgd, acc_linear_svc, acc_decision_tree]})
# print(models.sort_values(by='Score', ascending=False))
