# -------------------------------------------------------------------------
# AUTHOR: Aditya Dhar
# FILENAME: knn.py
# SPECIFICATION: Uses knn algorithm to make predictions
# FOR: CS 4200- Assignment #2
# TIME SPENT: 45 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# loop your data to allow each instance to be your test set
accuracy = 0
for i, instance in enumerate(db):
    # add the training features to the 2D array X and remove the instance that will be used for testing in this
    # iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    # transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the
    # instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert values to
    # float to avoid warning messages

    # --> add your Python code here
    X = []
    Y = []
    testSample = []

    for j in range(len(db)):
        row = []
        for k in range(len(db[j])):
            item = db[j][k]

            if k == 2:
                if j != i:
                    if item == '-':
                        Y.append(0.0)
                    else:
                        Y.append(1.0)
                else:
                    if item == '-':
                        row.append(0.0)
                    else:
                        row.append(1.0)
            else:
                row.append(float(item))

        if j != i:
            X.append(row)
        else:
            testSample = row
        # print(str(j) + ": " + row.__str__() + ", " + X.__str__())

    # print(X)
    # print(Y)
    # print(testSample)

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]

    actual = testSample[2]
    prediction = clf.predict([testSample[:2]])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # print("Prediction: " + str(prediction) + "; Actual: " + str(actual))
    if actual == prediction:
        accuracy += 1

# print the error rate
accuracy = accuracy / len(db)
print("Accuracy: " + str(accuracy))
