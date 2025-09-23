import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys

########### READ IN REF ##########
actual = {}
f = open('testdataActual.csv')
for line in f:
    parts = line.rstrip().split(",")
    actual[parts[0]] = parts[1]
f.close()

expected = []
predicted = []
f = open(sys.argv[1])
for line in f:
    parts = line.rstrip().split(",")
    filename = parts[0]
    if "wav" in filename:
        filename = re.sub("wav", "mp3", filename)
    if "mp3" not in filename:
        filename = filename + ".mp3"
    actualdx = actual[filename]
    predicteddx = parts[1]
    predicted.append(predicteddx)
    expected.append(actualdx)
f.close()



# Print some output
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
