# -------------------------------------------------------------------------
# AUTHOR: Aditya Dhar
# FILENAME: naive_bayes.py
# SPECIFICATION: Uses naive bayes algorithm to make predictions
# FOR: CS 4200- Assignment #2
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
dbTraining = []
X = []
Y = []

# reading the training data in a csv file
with open("weather_training.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining.append(row)

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
# X =

itemsIndex = [{"Sunny": 1, "Overcast": 2, "Rain": 3}, {"Hot": 1, "Mild": 2, "Cool": 3},
                  {"High": 1, "Normal": 2}, {"Strong": 1, "Weak": 2}, {"Yes": 1, "No": 2}]

for row in dbTraining:
    matrixRow = []
    for i in range(5):
        if i > 0:
            matrixRow.append(float(itemsIndex[i - 1][row[i]]))
    X.append(matrixRow)

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

resultIndex = {"Yes": 1, "No": 2}
for row in dbTraining:
    Y.append(float(resultIndex[row[5]]))

print(X)
print(Y)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
dbTest = []

with open("weather_test.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for j, row in enumerate(reader):
        if j > 0:  # skipping the header
            matrixRow = []
            for k in range(len(row)):
                if 0 < k < 5:
                    matrixRow.append(float(itemsIndex[k - 1][row[k]]))
            dbTest.append(matrixRow)

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(
    15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for i, test in enumerate(dbTest):
    prediction = clf.predict_proba(([test[:4]]))[0]
    PYes = prediction[0]
    PNo = prediction[1]

    predictionText = ""

    if PYes > PNo:
        predictionText = "Yes".ljust(15) + str(PYes).ljust(15)
    else:
        predictionText = "No".ljust(15) + str(PNo).ljust(15)

    # printing
    print(str(i + 15).ljust(15) + str(test[0]).ljust(15) + str(test[1]).ljust(15) + str(test[2]).ljust(15)
          + str(test[3]).ljust(15) + predictionText)
