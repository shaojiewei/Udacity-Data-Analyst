#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#==============================================================================
# print "data:", data
# print "labels:", labels
# print "features:", features
#==============================================================================
### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from time import time

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)
clf = DecisionTreeClassifier()
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3)

t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time() - t0, 3)

print accuracy_score(labels_test, pred)
#==============================================================================
# from sklearn.cross_validation import train_test_split
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.metrics import accuracy_score
# from time import time
# 
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# 
# clf = DecisionTreeClassifier()
# 
# t0 = time()
# clf.fit(features, labels)
# trainingTime = round(time()-t0, 4)
# 
# t0 = time()
# predictions = clf.predict(features_test)
# predictionTime = round(time()-t0, 4)
# 
# acc_score = accuracy_score(labels_test, predictions)
# 
# print "trainingTime: ", trainingTime
# print "predictionTime: ", predictionTime
# print "clf score: ", clf.score(features_test, labels_test)
# print "acc score: ", acc_score
# 
#==============================================================================
