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

from sklearn import preprocessing

train_df = pd.read_csv('datasetCopy.csv')
Paths = train_df["path"]
Labels = train_df["code"]


Paths = list(Paths)
Labels = list(Labels)
Features =[]
for path in Paths:

    img = cv2.imread(path)
    # print(img.shape)
    img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, bw_img = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY) #convert to binary
    black_char = cv2.bitwise_not(bw_img) #back to black char
    resized = cv2.resize(black_char,(28,28))
    Feature = np.reshape(resized, 784)
    Features.append(Feature)
    # cropped_img = crop_image(black_char)
    # Features.append(getFeatureVector(cropped_img)) 
print(len(Features))
print(len(Labels))
folder ='savedmodels/trial2/'

# Support Vector Machines
svc = SVC()
svc.fit(Features, Labels)
# Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(Features, Labels) * 100, 2)
print(acc_svc)
filename = folder+'SVM'+str(acc_svc)+'.sav'
pickle.dump(svc, open(filename, 'wb'))


#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(Features, Labels)
# Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(Features, Labels) * 100, 2)
print(acc_knn)
filename = folder+'knn'+str(acc_knn)+'.sav'
pickle.dump(knn, open(filename, 'wb'))

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(Features, Labels)
# Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(Features, Labels) * 100, 2)
print(acc_gaussian)
filename = folder+'Naive_Bayes'+str(acc_gaussian)+'.sav'
pickle.dump(gaussian, open(filename, 'wb'))



# Perceptron
perceptron = Perceptron(tol= not None)
perceptron.fit(Features, Labels)
# Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(Features, Labels) * 100, 2)
print(acc_perceptron)
filename = folder+'perceptron'+str(acc_perceptron)+'.sav'
pickle.dump(perceptron, open(filename, 'wb'))

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(Features, Labels)
# Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(Features, Labels) * 100, 2)
print(acc_linear_svc)
filename = folder+'linear_svc'+str(acc_linear_svc)+'.sav'
pickle.dump(linear_svc, open(filename, 'wb'))

# Stochastic Gradient Descent
sgd = SGDClassifier(tol= not None)
sgd.fit(Features, Labels)
# Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(Features, Labels) * 100, 2)
print(acc_sgd)
filename = folder+'sgd'+str(acc_sgd)+'.sav'
pickle.dump(sgd, open(filename, 'wb'))

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(Features, Labels)
# Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(Features, Labels) * 100, 2)
print(acc_decision_tree)
filename = folder+'decision_tree'+str(acc_decision_tree)+'.sav'
pickle.dump(decision_tree, open(filename, 'wb'))


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(Features, Labels)
# Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(Features, Labels) * 100, 2)
print(acc_random_forest)
filename = folder+'random_forest'+str(acc_random_forest)+'.sav'
pickle.dump(random_forest, open(filename, 'wb'))

