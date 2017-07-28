#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from tester import test_classifier

import matplotlib.pyplot as plt


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

label = 'poi'
financial_features = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]
email_features = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
]
features_list = [label] + financial_features + email_features
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Explore dataset
print "Total number of data points:" ,len(data_dict.keys())
print "Number of POIs:", sum(data_dict[name]['poi'] for name in data_dict.keys())

missing_values = dict(zip(features_list, [0 for _ in features_list]))
print missing_values
for feature in features_list:
    for name in data_dict.keys():
        if data_dict[name][feature] == 'NaN':
            missing_values[feature] += 1

print "Missing values by feature:"
for feature in features_list:
    print feature," ",str(missing_values[feature])
#print data_dict
### Task 2: Remove outliers
# Explore outliers

# 定义绘制散点图函数
def plotScatter(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red"""
    data = featureFormat(data_dict, [feature_x, feature_y, label])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            col = 'red'
        else:
            col = 'green'
        plt.scatter(x, y, color = col)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


    
plotScatter(data_dict, 'total_payments', 'total_stock_value')
plotScatter(data_dict, 'salary', 'bonus')
plotScatter(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')

# 移除异常值
records_to_remove = ['LOCKHART EUGENE E', 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for record in records_to_remove:
    data_dict.pop(record, 0)



        
### Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages):
    """
    """
    if poi_messages != "NaN" or all_messages != "NaN":
        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0.
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

features_new = ["fraction_from_poi", 'fraction_to_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = features_list + features_new

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transfrom(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# 4.1 Logistic Regression Classifier
form sklearn.linear_model import LogisticRegression

l_clf = Pipeline(steps=[
        ('sclaer', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = '12',random_state = 42))])


from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier()
clf_tree.fit(features, labels)
unsorted_pairs = zip(features_list[1:], clf_tree.feature_importances_)
sorted_dict = sorted(
        unsorted_pairs, key = lambda feature: feature[1], reverse = True)
tree_best_features = sorted_dict[:len(features_list)]
print tree_best_features

print "---Sorted Features from Decison Tree Feature Importancew---"
features_list = ['poi']
for item in tree_best_features:
    # print item[0], item[1]
    if item[1] > 0.1:
        features_list.append(item[0])
print features_list

# Extract features and labels from dataset with new features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

# DecisionTree
params_tree = {"min_samples_split": [2, 5, 10, 20],
               "criterion": ('gini', 'entropy')}

clf_tree = GridSearchCV(clf_tree, params_tree, cv = cv)
clf_tree.fit(features, labels)

print("--Decision Tree--")
clf_tree = clf_tree.best_estimator_
test_classifier(clf_tree, my_dataset, features_list)


#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_tree, my_dataset, features_list)
