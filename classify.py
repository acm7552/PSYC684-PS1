import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
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

npdatafeaturesubset = npdata[:,[1,5,6,7,8]]

## see how many features and training examples you have
print("You have ", npdatafeaturesubset.shape[0], "training instances")
print("You have ", npdatafeaturesubset.shape[1], "features")

## classify with just those features
scores = cross_val_score(gnb, npdatafeaturesubset, nptarget, cv=5, scoring='f1')
print("Baseline classification F1:", np.average(scores))


########### CLASSIFICATION ON A HELD-OUT SET  ############

# randomly select a subset of your data (size = 10)
testid = [1, 1]
while len(testid) != len(set(testid)):
    testid = np.random.randint(0, npdata.shape[0], 10)
    print(testid)

# Get your testing data
print(testid)
testset = npdata[testid, :]
testtarget = nptarget[testid]
print(testset.shape)

# Get your training data
trainset = np.delete(npdata, testid, 0)
traintarget = np.delete(nptarget, testid, 0)
print(trainset.shape)

# Build model
model = GaussianNB()
model.fit(trainset, traintarget)

# Apply model to test set
expected = testtarget
predicted = model.predict(testset)

# Print some output
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
