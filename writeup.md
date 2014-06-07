Prediction of Human Activity using Accelerometer Data
========================================================

# Data preparation

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

preModel <- preProcess(training[, num_features_idx], method = c("knnImpute"))

ptraining <- cbind(training$classe, predict(preModel, training[, num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[, num_features_idx]))
prtesting <- predict(preModel, raw_testing[, num_features_idx])

# Fix Label on classe
names(ptraining)[1] <- "classe"
names(ptesting)[1] <- "classe"
```


# Model

Using a random forest model provides good enough accuracy to predict the twenty test cases.

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf_model <- randomForest(classe ~ ., ptraining)
```


# Cross Validation

## In sample accuracy

```r
training_pred <- predict(rf_model, ptraining)
print(table(training_pred, ptraining$classe))
```

```
##              
## training_pred    A    B    C    D    E
##             A 5022    0    0    0    0
##             B    0 3418    0    0    0
##             C    0    0 3080    0    0
##             D    0    0    0 2895    0
##             E    0    0    0    0 3247
```

```r
print(mean(training_pred == ptraining$classe))
```

```
## [1] 1
```


## Out of sample accuracy

```r
testing_pred <- predict(rf_model, ptesting)
print(table(testing_pred, ptesting$classe))
```

```
##             
## testing_pred   A   B   C   D   E
##            A 556   2   0   0   1
##            B   1 374   1   0   0
##            C   0   2 338   5   0
##            D   0   0   3 315   0
##            E   1   1   0   1 359
```

```r
print(mean(testing_pred == ptesting$classe))
```

```
## [1] 0.9908
```


# Results


```r
answers <- predict(rf_model, prtesting)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

