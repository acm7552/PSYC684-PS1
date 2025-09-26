import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics, tree
from sklearn.preprocessing import normalize, StandardScaler
import sys

########### READ IN THE DATA ##########

labels = []
f = open(sys.argv[1])

# read in the first line as the feature labels
# so you can know what you're removing or adding
labels = f.readline().rstrip().split(",")

## read features into list of lists (data)
## read diagnosis into a list of lists (target)
target = []
data = []
print("The features are:", labels)
for line in f:
    parts = line.split(",")
    data.append([float(p) for p in parts[0:-1]])
    target.append(float(parts[-1]))

## conver to numpy arrays
nptarget = np.array(target)
npdata = np.array(data)


## normalize the columns
## npdata = normalize(data, norm='l2', axis=1)


########### CLASSIFICATION WITH CROSS VALIDATION  ############

## see how many features and training examples you have
print("You have ", npdata.shape[0], "training instances")
print("You have ", npdata.shape[1], "features")



## very, very basic classification with Naive Bayes classifier
gnb = GaussianNB()
scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
print("Baseline classification F1:", np.average(scores))

########### USING A SUBSET OF THE FEATURES #########
## Let's say you only want to use features 3 and 4...
# npdatafeaturesubset = npdata[:,3:5]
subsetArray = [1,3,4,5,6,7,8,9,10]
# subsetArray = [1,3, 7]
npdatafeaturesubset = npdata[:,subsetArray]

## see how many features and training examples you have
print("You have ", npdatafeaturesubset.shape[0], "training instances")
print("You have ", npdatafeaturesubset.shape[1], "features")
featureLabels = np.array(labels)
featureIndices = np.array(subsetArray)
print(f"feature(s): {featureLabels[featureIndices]}")

## classify with just those features
scores = cross_val_score(gnb, npdatafeaturesubset, nptarget, cv=5, scoring='f1')
print("Baseline classification F1:", np.average(scores))


########### CLASSIFICATION ON A HELD-OUT SET  ############

# randomly select a subset of your data (size = 10)
testid = [1, 1]
while len(testid) != len(set(testid)):
    testid = np.random.randint(0, npdatafeaturesubset.shape[0], 10)
    print(testid)

# Get your testing data
print(testid)
testset = npdatafeaturesubset[testid, :]
testtarget = nptarget[testid]
print(testset.shape)

# Get your training data
trainset = np.delete(npdatafeaturesubset, testid, 0)
traintarget = np.delete(nptarget, testid, 0)
print(trainset.shape)


# try scandard scaler for the data, since the nn is not liking unscaled data
scaler = StandardScaler()
trainset = scaler.fit_transform(trainset)
testset = scaler.transform(testset)


# Build model
# model = SVC(kernel="linear")
#model = tree.DecisionTreeClassifier()
#model = GaussianNB()
model = MLPClassifier(hidden_layer_sizes=(12, 4), 
                            activation='relu',                           
                            learning_rate_init=1e-2,
                            solver="adam",
                            max_iter=400,
                            random_state=42,
                            verbose=True)


model.fit(trainset, traintarget)



# Apply model to test set
expected = testtarget
predicted = model.predict(testset)
#print(predicted)
#print(model.predict_proba(testset))
#print(model.predict_proba(trainset))

# Print some output
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
