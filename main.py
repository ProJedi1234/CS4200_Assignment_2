# -------------------------------------------------------------------------
# AUTHOR: Aditya Dhar
# FILENAME: main.py
# SPECIFICATION: Runs several decision trees to find the precision of the models
# FOR: CS 4200- Assignment #2
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
setNumber = 0
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    # X =

    itemsIndex = [{"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3}, {"Myope": 1, "Hypermetrope": 2},
                  {"Yes": 1, "No": 2}, {"Normal": 1, "Reduced": 2}]

    for row in dbTraining:
        matrixRow = []
        for i in range(4):
            matrixRow.extend([itemsIndex[i][row[i]]])
        X.append(matrixRow)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> addd your Python code here
    # Y =

    resultIndex = {"Yes": 1, "No": 2}
    for row in dbTraining:
        Y.append(resultIndex[row[4]])

    # loop your training and test tasks 10 times here
    lowestAccuracy = 1
    for i in range(10):
        # print("RUN " + str(i + 1))
        accuracy = 0

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        dbTest = []

        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:  # skipping the header
                    dbTest.append(row)

        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during
            # training, and then use the decision tree to make the class prediction. For instance: class_predicted =
            # clf.predict([[3, 1, 2, 1]])[0] where [0] is used to get an integer as the predicted class label so that
            # you can compare it with the true label --> add your Python code here

            # compare the prediction with the true label (located at data[4]) of the test instance to start
            # calculating the accuracy. --> add your Python code here
            testRow = []
            for k in range(4):
                testRow.extend([itemsIndex[k][data[k]]])
            prediction = clf.predict([testRow])[0]
            answer = resultIndex[data[4]]

            if answer == prediction:
                accuracy += 1

            # print("Prediction: " + str(prediction) + "; Actual: " + str(answer))

        accuracy = accuracy / 8
        # print("Accuracy: " + str(accuracy))
        if accuracy < lowestAccuracy:
            lowestAccuracy = accuracy
            # print("Lowest: " + str(lowestAccuracy))

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here

    print("final accuracy when training on " + ds + ": " + str(accuracy))
