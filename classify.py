#import re
import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics, tree
from sklearn.preprocessing import normalize, StandardScaler
from scipy.interpolate import interp1d
#import sys

########### READ IN THE DATA ##########

#labels = []
#f = open(sys.argv[1])

# read in the first line as the feature labels
# so you can know what you're removing or adding
#labels = f.readline().rstrip().split(",")

## read features into list of lists (data)
## read diagnosis into a list of lists (target)
target = []
data = []

df = pd.read_csv("dialectsdataoutput.csv")

data = df.iloc[:, :-1]
target = df.iloc[:, -1]

labels = data.columns

print("The features are:", labels)

## can also grab other engineered features here like ratio of f1 to f2

#data["f1/f2"] = data["meanf1"] / data["meanf2"]
## CHANGE FEATURES FOR USING HERE
chosenFeatures = ["pitch", "meanf1", "meanf2"]
#chosenFeatures = []
print("Chosen features: ", chosenFeatures)
# for line in f:
#     parts = line.split(",")
#     data.append([float(p) for p in parts[0:-1]])
#     target.append(float(parts[-1]))


# use this w/ np.apply to get the mfcc coefficients and then resize them
def getMFCC(row):
    filename = f"MFCC/{row}MFCC.csv"
    mfcc = pd.read_csv(filename, header=None, sep='\s+')
    #print(mfcc.shape)

    mfcc_np = mfcc.to_numpy()
    coefficients, frames = mfcc_np.shape # 12 * 2200 or something, frames is not the same

    # old and new frameIndex 
    oldIndex = np.arange(frames)
    newIndex = np.linspace(0, frames - 1, 25) 

    # rows = coefficients, columns = frames
    mfcc_resized = np.zeros((coefficients, 25))
    for c in range(coefficients):
        f = interp1d(oldIndex, mfcc_np[c, :], kind='linear')
        mfcc_resized[c, :] = f(newIndex)

    # should be 12 by 25
    #print(mfcc_resized.shape) 
    mfcc_resized_flattened = mfcc_resized.flatten()
    #print(mfcc_resized_flattened.shape)
    return mfcc_resized_flattened



## convert to numpy arrays
nptarget = np.array(target)
npdata = np.array(data[chosenFeatures])


usingMFCC = True
if usingMFCC:
    data["mfcc"] = data["mfcc"].apply(getMFCC)
    #print(data)
    npMFCC = np.stack(data["mfcc"].to_numpy())
    #print(npMFCC.shape)
    #npdata = np.concatenate(npdata, np.array(data["mfcc"]))
    npdata = np.hstack([npdata, npMFCC])
    #print(npdata.shape)


## normalize the columns
## npdata = normalize(data, norm='l2', axis=1)


########### CLASSIFICATION WITH CROSS VALIDATION  ############

## see how many features and training examples you have
print("You have ", npdata.shape[0], "training instances")
print("You have ", npdata.shape[1], "features")


       
## very, very basic classification with Naive Bayes classifier
#gnb = GaussianNB()
#scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
#print("Baseline classification F1:", np.average(scores))

########### USING A SUBSET OF THE FEATURES #########
## Let's say you only want to use features 3 and 4...
# npdatafeaturesubset = npdata[:,3:5]
#subsetArray = [1,3,4,5,6,7,8,9,10]
# subsetArray = [1,3, 7]
#npdatafeaturesubset = npdata[:,subsetArray]

## see how many features and training examples you have
#print("You have ", npdatafeaturesubset.shape[0], "training instances")
#print("You have ", npdatafeaturesubset.shape[1], "features")
#featureIndices = np.array(subsetArray)
#print(f"feature(s): {featureLabels[featureIndices]}")

## classify with just those features
#scores = cross_val_score(gnb, npdatafeaturesubset, nptarget, cv=5, scoring='f1')
#print("Baseline classification F1:", np.average(scores))


########### CLASSIFICATION ON A HELD-OUT SET  ############


# seed = random.randint(1, 10000)
seed = 42
print("seed: ", seed)
# try scandard scaler for the data, since the nn is not liking unscaled data
#scaler = StandardScaler()
#X = scaler.fit_transform(npdata)
X = npdata
y = nptarget
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=seed, shuffle=True)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# randomly select a subset of your data (size = 10)
# testid = [1, 1]
# while len(testid) != len(set(testid)):
#     testid = np.random.randint(0, npdatafeaturesubset.shape[0], 10)
#     print(testid)

# get testing data
# print(testid)
# testset = npdatafeaturesubset[testid, :]
# testtarget = nptarget[testid]
# print(testset.shape)

# get training data
# trainset = np.delete(npdatafeaturesubset, testid, 0)
# traintarget = np.delete(nptarget, testid, 0)
# print(trainset.shape)


# try scandard scaler for the data, since the nn is not liking unscaled data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build model
# model = SVC(kernel="linear")
#model = tree.DecisionTreeClassifier()
#model = GaussianNB()
model = MLPClassifier(hidden_layer_sizes=(400, 100, 48, 12), 
                            activation='relu',                           
                            learning_rate_init=1e-2,
                            solver="adam",
                            max_iter=50,
                            random_state=seed,
                            verbose=True)


model.fit(X_train, y_train)



# Apply model to test set
expected = y_test
predicted = model.predict(X_test)
#print(predicted)
#print(model.predict_proba(testset))
#print(model.predict_proba(trainset))

# Print some output
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
