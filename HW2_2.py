import csv

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import cross_validation
import random

def Randomlist(k, lenth_all):
    listrandom = random.sample(range(lenth_all), k)
    return listrandom

def RelistData(k):
    csvfile = open(k, newline='')
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='\n')
    data = list(csvreader)
    m = data[0]
    data.remove(m)
    line = 0
    return data




def RunPredict(k,datasize):

    listA = []
    listB = []
    line = 0
    datafile = '2013.csv' #get data

    data_input = RelistData(datafile)
    #single file
    data_train = data_input
    for line in range(len(data_train)):
        listA.append([data_train[line][2]  , data_train[line][3], data_train[line][4], data_train[line][5] , data_train[line][8], data_train[line][11] ]) #, data_train[line][9] , data_train[line][10]
        listB.append(data_train[line][6])
        line += 1



    #print (len(listA))
    X = np.array(listA)
    Y = np.array(listB)

# fit a SVM model to the data
    print("Training Time:")
    print(time.strftime("%m/%d  %H:%M:%S")) ##24小时格式
    model = SVC()

    model.fit(X, Y)
    print(time.strftime("%m/%d  %H:%M:%S"))  ##24小时格式
    print(model)
    print("=================")
    # make predictions
    expected = Y

    datatestString = '2014.csv'
    data_test = RelistData(datatestString)
    random_list_test = Randomlist(datasize, len(data_test))
    randomrow_test = 0
    list_test = []
    list_ans = []
    line_test = 0
    #for line_test in range(len(data_test)):

    for line_test in range(datasize):
        randomrow_test = random_list_test[line_test]
        list_test.append([data_test[randomrow_test][2]  , data_test[randomrow_test][3], data_test[randomrow_test][4], data_test[randomrow_test][5] ,data_test[randomrow_test][8],  data_test[randomrow_test][11] ]) # data_test[randomrow_test][9] , data_test[randomrow_test][10]
        list_ans.append(data_test[randomrow_test][6])
        line_test += 1

    predicted = model.predict(list_test)

    '''''''''
    dataOutString = 'OUT' + str(k) + '.csv'

    f = open(dataOutString, 'w', newline='')
    w = csv.writer(f)
    data_output = [[]]
    data_output.append(predicted)
    w.writerows(data_output)
    f.close()
    '''''''''
    print("Test Time:")
    print(time.strftime("%m/%d  %H:%M:%S"))  ##24小时格式
    #print(predicted)

    print(time.strftime("%m/%d  %H:%M:%S"))  ##24小时格式
    print("=================")

    print(model.score(X, Y, sample_weight=None))


    X2= np.array(list_test)
    Y2 = np.array(list_ans)
    print(model.score(X2, Y2, sample_weight=None))
      #print(predicted)

    corectAns = 0
    check_no = 0
    print(time.strftime("%m/%d  %H:%M:%S"))  ##24小时格式
    for check_no in range(len(list_ans)):
        l = 0

        for l in range(len(data_input)):
            if data_input[l][0] == data_test[check_no][0]:
                if data_input[l][2] == data_test[check_no][2]:
                    if data_input[l][8] == data_test[check_no][8]:
                        if data_input[l][6] == predicted[check_no]:
                            corectAns+=1
                            break
    print(time.strftime("%m/%d  %H:%M:%S"))  ##24小时格式
    print(corectAns)
    print(100*corectAns/len(list_ans))

TEST = [1000,2000,5000,7000,10000,15000]
for i in range (len(TEST)):
    j = TEST[i]
    print("Training size: "+ str(j))
    for k in range (1,6) :
        print(k)
        RunPredict(k,j)


