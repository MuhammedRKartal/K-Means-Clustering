#Author: Muhammed Rahmetullah Kartal
#This code compares a written KMeans algorithm with SKLearn library's Kmeans algorithm
#and plots them.

import pandas as pd
import math
from random import randint as ri

#plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans #sklearn Kmeans clustering function
import warnings
warnings.filterwarnings("ignore")


#calculates euclidean distance between 2 dataframe elements
def euclideanDistance(main, compare):
    return math.sqrt((compare.y - main.y)**2 +(compare.x - main.x)**2)

#helper method for calculating objective function threshold
def diffL2(values):
    diff = abs(values[-2] - values[-1])
    return diff

#Objective function that is sum of distances of each point to its cluster center
def objFunction(data,sp,k):
    sum = 0
    for i in range(data.shape[0]):
        #this loop is to find correct cluster center and calculating distance
        for j in range(1,k+1):
            if data['p'].loc[i] == j : #finds the correct cluster center
                sum += euclideanDistance(data.loc[i],sp[j-1]) #calculates distance
    return sum

#K random initial cluster centers on given dataframe
#returns array of pandas series
def selectStartingPoints(data,k):
    starterList = list()
    while k!=0:
        place = ri(0,data.shape[0])
        starterList.append(data.loc[place])
        k-=1
    return starterList


#Finding K cluster centers related to mean of points
#returns array of pandas series
def findMeanPoints(data, k):
    #Creating a K lenght array
    meansList = list()
    for i in range(k): meansList.append([0, 0])

    counts = [0] * k #storing count of classes for math operations
    spList = list() #list to store pandas series, stores K points

    #This loops calculates means related to classes part 1(summing)
    for i in range(data.shape[0]):
        for j in range(1, k + 1): #1, k+1 because classes starts from 1
            if data['p'].loc[i] == j:
                meansList[j - 1][0] += data.loc[i].x
                meansList[j - 1][1] += data.loc[i].y
                counts[j - 1] += 1

    #mean part 2(dividing)
    for i in range(k):
        meansList[i][0] /= counts[i]
        meansList[i][1] /= counts[i]

    #creating dictionary and a dataframe
    d = {'x': [x[0] for x in meansList], 'y': [x[1] for x in meansList], 'c': None, 'p': None}
    d = pd.DataFrame(data=d)

    #storing series into a list
    for i in range(d.shape[0]):
        spList.append(d.loc[i])
    return spList


#finds correct class for given point by checking distances to cluster centers
def findClass(current, sp):
    dList = list()
    for i in range(len(sp)):
        dList.append(euclideanDistance(current, sp[i]))#distance calculation

    #finding min distance and returning correct class
    m = min(dList)
    for i in range(len(sp)):
        if dList[i] == m:
            return i + 1

#Predictions iteration
def oneIteration(data,sp):
    for i in range(data.shape[0]):
        data['p'].loc[i] = findClass(data.loc[i],sp)
    return data

#Main method to run Kmeans and plotting.
#Returns predicted dataframe, and objective function values
def kMeans(data, k, T):
    #giving names to dataframe columns, and creating a new column named 'p' which is predictions
    #'x' is x axis coordinates, 'y' is y axis coordinates, 'c' is class
    data.columns = ['x', 'y', 'c'] if len(data.columns) == 3 else ['x', 'y', 'c', 'p']
    data['p'] = None

    #counting iterations and objective function values to graph them
    iterCount = 0
    objectiveFunctions = list()


    #creating initial starter points
    sp = selectStartingPoints(data, k)
    spDF = pd.DataFrame(sp)

    #running one iteration and calculating objective function
    data = oneIteration(data, sp)
    val = objFunction(data, sp, k)
    objectiveFunctions.append((iterCount, val))

    #plotting initial stage
    sns.scatterplot(data=data, x=data.x, y=data.y, hue=data.p)
    sns.scatterplot(data=spDF, x=spDF.x, y=spDF.y, s=200, color='g', marker='X')
    plt.show()

    #this loop calculates mean points and plots each iteration until
    #differences between consecutive objective functions is lesser than Threshold value
    while True:
        iterCount += 1
        plt.clf() #clearing the plot screen

        sp = findMeanPoints(data, k)
        spDF = pd.DataFrame(sp)

        data = oneIteration(data, sp)
        val = objFunction(data, sp, k)
        objectiveFunctions.append((iterCount, val))

        sns.scatterplot(data=data, x=data.x, y=data.y, hue=data.p)
        sns.scatterplot(data=spDF, x=spDF.x, y=spDF.y, s=200, color='g', marker='X')

        plt.show()
        #threshold check
        if diffL2([v[1] for v in objectiveFunctions]) < T:
            break

    return data, objectiveFunctions


#Running sklearn kmeans algorithm
def runSKLearn(data, k):
    data.columns = ['x', 'y', 'c'] if len(data.columns) == 3 else ['x', 'y', 'c', 'p']
    data['p'] = None

    dataArr = data.drop(['c', 'p'], axis=1).to_numpy() #sklearn uses numpy array to train
    model = KMeans(n_clusters=k)
    model.fit(dataArr)

    data.p = model.predict(dataArr) #putting prediction array into dataframe

    centers = pd.DataFrame(model.cluster_centers_, columns=['x', 'y']) #getting center points as dataframe

    #plotting final version
    plt.clf()
    sns.scatterplot(data=data, x=data.x, y=data.y, hue=data.p)
    sns.scatterplot(data=centers, x=centers.x, y=centers.y, s=200, color='g', marker='X')
    plt.show()

#Plotting the Objective function values-Iteration Count graph by using lineplot and scatterplot
def drawObjectiveFunctionGraph(data):
    ax = sns.lineplot(data=data, x= data.iterationCount, y=data.objectiveFunction)
    ax = sns.scatterplot(data=data, x= data.iterationCount, y=data.objectiveFunction)
    ax.set(xLabel="Iteration Count", yLabel="Objective Function Value", title= "Objective Function")
    plt.show()
    return ax

#reading data from txt as dataframe
data = pd.read_csv('data3.txt', header=None)

#running my algorithm
data,objectiveFunctions = kMeans(data,3,0.1)

#running sklearn algorithm
runSKLearn(data,3)

#creating dataframe for objective function values-iteration count graph and plotting it
graphData = pd.DataFrame(objectiveFunctions, columns = ['iterationCount','objectiveFunction'])
drawObjectiveFunctionGraph(graphData)