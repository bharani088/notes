>ref: ***Machine Learning in Action***

# CLASSIFICATION

## k-Nearest Neighbors

>Pros: High accuracy, insensitive to outliers, no assumptions about data 
>Cons: Computationally expensive, requires a lot of memory
>Works with: Numeric values, nominal values

General approach to kNN
1. Collect: Any method.
2. Prepare: Numeric values are needed for a distance calculation. A structured data format is best.
3. Analyze: Any method.
4. Train: Does not apply to the kNN algorithm.
5. Test: Calculate the error rate.
6. Use: This application needs to get some input data and output structured num- eric values. Next, the application runs the kNN algorithm on this input data and determines which class the input data should belong to. The application then takes some action on the calculated class.

```python
from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  diffMat = tile(inX, (dataSetSize,1)) - dataSet
  sqDiffMat = diffMat**2
  sqDistances = sqDiffMat.sum(axis=1)
  distances = sqDistances**0.5
  sortedDistIndicies = distances.argsort()     
  classCount={}          
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
  # sort in book
  # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
  # sort a dict by keys in python2.4 or greater, then use classCount[key]
  # sortedkeylist = sorted(classCount.iterkeys())
  # sort a dict by value in python2.4 or greater
  sortedClassCount = sorted(classCount.iteritems(), key=lambda (k,v): (v,k), reverse=True)
  return sortedClassCount[0][0]

if __name__ == '__main__':
  dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
  labels = ['A','A','B','B']
  class = classify0([0,0], dataSet, labels, 3)
  print(class)
```

The k-Nearest Neighbors algorithm is a simple and effective way to classify data. 

kNN is an example of instance-based learning, where you need to have instances of data close at hand to perform the machine learning algorithm. The algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome.

An additional drawback is that kNN doesn’t give you any idea of the underlying structure of the data; you have no idea what an “average” or “exemplar” instance from each class looks like. In the next chapter, we’ll address this issue by exploring ways in which probability measurements can help you do classification.

## Decision Trees

## Naïve Bayes

## Logistic Regression

## Support Vector Machines

## AdaBoost meta-algorithm



# FORECASTING NUMERIC VALUES WITH REGRESSION

## Predicting numeric values: regression

## Tree-based regression



# UNSUPERVISED LEARNING

## Grouping unlabeled items using k-means clustering

## Association analysis with the Apriori algorithm

## Efficiently finding frequent itemsets with FP-growth