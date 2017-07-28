#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

data_dict.pop("TOTAL", 0) # remove outlier
data = featureFormat(data_dict, features)


### your code below
max_num = 0
name = 0
for point in data:
    salary = point[0]
    bonus = point[1]
    if bonus > max_num:
        max_num = bonus
        name = salary
    matplotlib.pyplot.scatter( salary, bonus )

print max_num
print name
for name1 in data_dict.keys():
    if data_dict[name1]["bonus"] != 'NaN' and  data_dict[name1]["salary"] != 'NaN':
        if data_dict[name1]["bonus"] > 5000000 and data_dict[name1]["salary"] > 1000000:
            print name1
#print data_dict["CORDES WILLIAM R"]
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

