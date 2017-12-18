Machine Learning
================
Jennifer Bochenek
December 17, 2017

Project Summary
===============

In this project I take in the movement data and try to create a model that will sucessfully predict the type of activity that the person is doing using machine learning methods. I walk through the steps of pre-processing the data, creating the data, and discuss the results. Finally, I use my model to predict the testing data for the final course quiz.

Setup workspace
---------------

``` r
require(caret)
```

    ## Loading required package: caret

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
require(parallel)
```

    ## Loading required package: parallel

``` r
require(doParallel)
```

    ## Loading required package: doParallel

    ## Loading required package: foreach

    ## Loading required package: iterators

Load in the data
----------------

Check if the files exist, if they don't then the code will download it. This code also turns invalid data into NA or other missing codes as appropriate.

``` r
target <- "pml_training.csv"
if (!file.exists(target)) {
    url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    target <- "pml_training.csv"
    download.file(url, destfile = target)
}
training <- read.csv(target, na.strings = c("NA","#DIV/0!",""))

target <- "pml_testing.csv"
if (!file.exists(target)) {
    url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url, destfile = target)
}
testing <- read.csv(target, na.strings = c("NA","#DIV/0!",""))
```

Clean the data
--------------

This part of the code will clean the data, using the nearZeroVar function from the package 'Caret' to remove values that are close to zero. This also removes all values that are NA or null data.

``` r
subTrain <- training[, names(training)[!(nearZeroVar(training, saveMetrics = T)[, 4])]]
subTrain <- subTrain[, names(subTrain)[sapply(subTrain, function (x)
        ! (any(is.na(x) | x == "")))]]
```

Start creating the training data
--------------------------------

This code will take out number of cases needed from the training dataset for the validation set, which will be used later. Technically using random forest method this is less important due to the fact that the random forests method uses bagging and random subspace. But I will be using the validation sample to check the work later.

``` r
inTrain <- createDataPartition(subTrain$classe, p = 0.6, list = FALSE)
subTraining <- subTrain[inTrain,]
subValidation <- subTrain[-inTrain,]
```

Create the predition model
--------------------------

Although there are other modeling methods, I chose to use random forests. I have a brand new high CPU computer, so I didn't actually need to use the parallel processing (which actually through a bunch of errors) but felt that it was helpful to build into the code in case I wanted to run it on another computer later. The program is also designed to only run once, as it saves its output separately so that it won't try to re-run the time consuming process every time I knit this rmarkdown.

``` r
model <- "./data/model.RData"
if (!file.exists(model)) {
    clust <- makeCluster(detectCores() - 1)
    registerDoParallel(clust)
    fit <- train(classe ~ ., method = "rf", data = subTraining)
    save(fit, file = "./data/model.RData")
    stopCluster(clust)
} else {
    load(file = "./data/model.RData", verbose = TRUE)
}
```

    ## Loading objects:
    ##   fit

Find out the accuracy
=====================

As it says on the tin, it's a fairly standard code to get the prediction and generate the confusion matrix. I'm not sure that I trust that the accuracy was 100%, but that is the value given.

``` r
predTrain <- predict(fit, subTraining)
confusionMatrix(predTrain, subTraining$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3348    0    0    0    0
    ##          B    0 2279    0    0    0
    ##          C    0    0 2054    0    0
    ##          D    0    0    0 1930    0
    ##          E    0    0    0    0 2165
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Find the accuracy of the validation set
---------------------------------------

The validation set also has an accuracy of 100%, so the model is working very well.

``` r
predValidation <- predict(fit, subValidation)
confusionMatrix(predValidation, subValidation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232    0    0    0    0
    ##          B    0 1518    1    0    0
    ##          C    0    0 1367    0    0
    ##          D    0    0    0 1286    0
    ##          E    0    0    0    0 1442
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 0.9999     
    ##                  95% CI : (0.9993, 1)
    ##     No Information Rate : 0.2845     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 0.9998     
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   0.9993   1.0000   1.0000
    ## Specificity            1.0000   0.9998   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   0.9993   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   0.9998   1.0000   1.0000
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1935   0.1742   0.1639   0.1838
    ## Detection Prevalence   0.2845   0.1936   0.1742   0.1639   0.1838
    ## Balanced Accuracy      1.0000   0.9999   0.9996   1.0000   1.0000

``` r
varImp(fit)
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 80)
    ## 
    ##                                 Overall
    ## X                              100.0000
    ## roll_belt                        6.9550
    ## pitch_forearm                    1.9964
    ## raw_timestamp_part_1             1.9778
    ## accel_belt_z                     1.6205
    ## roll_dumbbell                    0.9976
    ## num_window                       0.9317
    ## accel_forearm_x                  0.7184
    ## cvtd_timestamp02/12/2011 14:57   0.7013
    ## magnet_dumbbell_y                0.5793
    ## magnet_belt_y                    0.5417
    ## cvtd_timestamp30/11/2011 17:12   0.5028
    ## total_accel_belt                 0.4191
    ## yaw_belt                         0.3690
    ## cvtd_timestamp30/11/2011 17:11   0.3092
    ## pitch_belt                       0.2973
    ## magnet_belt_z                    0.2950
    ## pitch_dumbbell                   0.2704
    ## roll_forearm                     0.2627
    ## magnet_dumbbell_x                0.2344

Quiz answers
------------

I ran this to generate my answers for the final quiz. 20/20! Sucess! That's how I know it's working well.

``` r
predTesting <- predict(fit, testing)
predTesting
```

    ##  [1] A A A A A A A A A A A A A A A A A A A A
    ## Levels: A B C D E
