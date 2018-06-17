# IrisDataSet
Predicting the class of the flower based on available attributes using Logistic Regression.

This is probably the most versatile, easy and resourceful dataset in pattern recognition literature. 
Nothing could be simpler than the Iris dataset to learn classification techniques. 
If you are totally new to data science, this is your start line. The data has only 150 rows & 4 columns.

I have used Spark-ML library's Logisitic Regression and have split the data randomly into Training(60%), Cross Validation Set(20%),
Test Set (20%) and have programmed to get the best possible regularization parameter when looking for best regularization parameter manually.

I have also added bestModel function which makes use of CrossValidator Class of Spark ML Library and finds out the best regularization parameter and maxIterations. 

Get the Data Set here : https://archive.ics.uci.edu/ml/datasets/Iris

###### Run the program using the following command

``spark-submit --class com.ankit.IrisDataSet.LogResIrisDataSet IrisDataSet-0.0.1-SNAPSHOT.jar path/to/irisData.txt``
