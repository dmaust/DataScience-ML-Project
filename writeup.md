Prediction of Weight Lifting Style using Accelerometer Data
========================================================

## Introduction

With the availability of low cost accelerometers, there are many opportunities to measure human activities. One application of this is [measuring the proper form of weight lifting][wle]. In this paper we examine whether we can determine the weight lifting form using the accelerometer data collected.


## Data preparation

Load both datasets.

```r
raw_training <- read.csv("training.csv")
raw_testing <- read.csv("testing.csv")
```


Partition training data provided into two sets. One for training and one for cross validation.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(1234)
trainingIndex <- createDataPartition(raw_training$classe, list = FALSE, p = 0.9)
training = raw_training[trainingIndex, ]
testing = raw_training[-trainingIndex, ]
```


Remove indicators with near zero variance.

```r
library(caret)
nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]
```



Filter columns to only include numeric features and outcome. Integer and other non-numeric features can be trained to reliably predict values in the training file provided, but when used to predict values in the testing set provided, they lead to misclassifications.

```r
num_features_idx = which(lapply(training, class) %in% c("numeric"))
```


We then would like to impute missing values as many exist in our training data.


```r
preModel <- preProcess(training[, num_features_idx], method = c("knnImpute"))

ptraining <- cbind(training$classe, predict(preModel, training[, num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[, num_features_idx]))
prtesting <- predict(preModel, raw_testing[, num_features_idx])

# Fix Label on classe
names(ptraining)[1] <- "classe"
names(ptesting)[1] <- "classe"
```


## Model

We can build a random forest model using the numerical variables provided. As we will see later this provides good enough accuracy to predict the twenty test cases. Using [caret][caret], we can obtain the optimal mtry parameter of 32. This is a computationally expensive process, so only the optimized parameter is shown below.


```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf_model <- randomForest(classe ~ ., ptraining, ntree = 500, mtry = 32)
```


## Cross Validation

We are able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.

### In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining)
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5022    0    0    0    0
##          B    0 3418    0    0    0
##          C    0    0 3080    0    0
##          D    0    0    0 2895    0
##          E    0    0    0    0 3247
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

The in sample accuracy is 100% which indicates, the model does not suffer from bias.

### Out-of-sample accuracy

```r
testing_pred <- predict(rf_model, ptesting)
```


Confusion Matrix: 

```r
print(confusionMatrix(testing_pred, ptesting$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556   2   0   0   0
##          B   0 373   0   0   2
##          C   1   3 339   1   0
##          D   0   0   3 319   0
##          E   1   1   0   1 358
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.987, 0.996)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.984    0.991    0.994    0.994
## Specificity             0.999    0.999    0.997    0.998    0.998
## Pos Pred Value          0.996    0.995    0.985    0.991    0.992
## Neg Pred Value          0.999    0.996    0.998    0.999    0.999
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.173    0.163    0.183
## Detection Prevalence    0.285    0.191    0.176    0.164    0.184
## Balanced Accuracy       0.997    0.991    0.994    0.996    0.996
```


The cross validation accuracy is greater than 99%, which should be sufficient for predicting the twenty test observations. Based on the lower bound of the confidence interval we would expect to achieve a 98.7% classification accuracy on new data provided. 

One caveat exists that the new data must be collected and preprocessed in a manner consistent with the training data.

## Test Set Prediction Results

Applying this model to the test data provided yields 100% classification accuracy on the twenty test observations.

```r
answers <- predict(rf_model, prtesting)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


## Conclusion
We are able to provide very good prediction of weight lifting style as measured with accelerometers.

## References

[wle]: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

[caret]: Max Kuhn. Contributions from Jed Wing, Steve Weston, Andre Williams, Chris Keefer, Allan Engelhardt, Tony Cooper, Zachary Mayer and the R Core Team. Caret package.
