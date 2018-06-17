package com.ankit.IrisDataSet

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidatorModel





object LogResIrisDataSet {
  
  var regParams = Array(0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24)
  var accuracies : ListBuffer[Double] = new ListBuffer()
  def createBestModel(trainSet : DataFrame, cvSet : DataFrame) : LogisticRegressionModel={
    
    var maxAccuracy : Double= 0;
    var maxAccuracyIndex = 0;
    for(i <- 0 to regParams.length-1){
     val model = new LogisticRegression()
                     .setLabelCol("label")
                     .setPredictionCol("predictedLabel")
                     .setRegParam(regParams(i))
                     .setMaxIter(100)
                     .fit(trainSet);
    val evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("label")
                  .setPredictionCol("predictedLabel")
                  .setMetricName("accuracy")
     val cvSetOutput = model.transform(cvSet).toDF();
     //cvSetOutput.show();
     val accuracy = evaluator.evaluate(cvSetOutput)
     accuracies = accuracies.+=(accuracy)
     if(accuracy>maxAccuracy)
     {
       maxAccuracyIndex = i;
       maxAccuracy = accuracy;  
     }
       
    }
    val model = new LogisticRegression()
                     .setLabelCol("label")
                     .setPredictionCol("predictedLabel")
                     .setRegParam(regParams(maxAccuracyIndex))
                     .setMaxIter(100)
                     .fit(trainSet);
    for (acc<-accuracies)
      print(acc*100+"\t")
    println("Max Accuracy Achieved on CV Set: "+maxAccuracy*100+"%")
    println("Regularization parameter Chosen : "+regParams(maxAccuracyIndex))
    return model;
  }
  
  def bestModel(trainSet : DataFrame, cvSet : DataFrame) : CrossValidatorModel={
    val lr = new LogisticRegression().setLabelCol("label")
                              .setPredictionCol("predictedLabel")  
    val evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("label")
                  .setPredictionCol("predictedLabel")
                  .setMetricName("accuracy")
    val paramMaps = new ParamGridBuilder()
                          .addGrid(lr.regParam, regParams)
                          .addGrid(lr.maxIter, Array(100,200))
                          .build()
    val cv = new CrossValidator().setEstimator(lr)
                  .setEvaluator(evaluator)
                  .setEstimatorParamMaps(paramMaps)
    val set = trainSet.union(cvSet);
    val model = cv.fit(set)
    return model;
  }
  
  def main(args : Array[String]) {
    val spark = SparkSession.builder().getOrCreate();
     val file = spark.read.text(args(0));
     val newNames = Seq("features","label")
     val seqData = file.rdd.map(row=>{
       val line = row.getString(0).split(",");
       val x1 = line(0).toDouble
       val x2 = line(1).toDouble
       val x3 = line(2).toDouble
       val x4 = line(3).toDouble
       val label = line(4) match{
                     case "Iris-setosa"=>1          
                     case "Iris-versicolor"=>2
                     case "Iris-virginica"=>3
                     }
      // println(x1+"\t"+x2+"\t"+x3+"\t"+x4+"\t"+label) 
       (Vectors.dense(1,x1,x2,x3,x4),label)
            
     });         
                     
     val df = spark.createDataFrame(seqData).toDF(newNames:_*);
     
     val Array(trainSet, cvSet, testSet)= df.randomSplit(Array(0.6,0.2,0.2));
     trainSet.persist()
     cvSet.persist()
     val model = bestModel(trainSet, cvSet)
     
     val evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("label")
                  .setPredictionCol("predictedLabel")
                  .setMetricName("accuracy")
     val testSetOutput = model.transform(testSet).toDF();
     testSetOutput.show();
     val accuracy = evaluator.evaluate(testSetOutput)
     println("Test set accuracy = " + accuracy*100+"%")  
  }

}
