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
traintarget = np.array(target)
trainset = np.array(data)

## now do the same for test data
## read diagnosis into a list of lists (target)
f = open(sys.argv[2])
testtargetlist = []
testdatalist = []
labels = f.readline().rstrip().split(",")
print("The features are:", labels)
for line in f:
    parts = line.split(",")
    testdatalist.append([float(p) for p in parts[0:-1]])
    testtargetlist.append(float(parts[-1]))

## conver to numpy arrays
testtarget = np.array(testtargetlist)
testset = np.array(testdatalist)



########### CLASSIFICATION WITH CROSS VALIDATION  ############

## see how many features and training examples you have
print("You have ", trainset.shape[0], "training instances")
print("You have ", trainset.shape[1], "features")

# Build model
model = GaussianNB()
model.fit(trainset, traintarget)

# Apply model to test set
expected = testtarget
predicted = model.predict(testset)

# Print some output
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
