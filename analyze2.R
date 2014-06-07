library(caret)
library(kernlab)

raw_training <- read.csv('training.csv')
raw_testing <- read.csv('testing.csv')

set.seed(1234)
trainingIndex <- createDataPartition(raw_training$classe, list=FALSE, p=.9)
training = raw_training[trainingIndex,]
testing = raw_training[-trainingIndex,]

nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]

training <- training[-5]
testing <- testing[-5]
raw_testing <- raw_testing[-5]

num_features_idx = which(lapply(training,class) %in% c('numeric')  )

preModel <- preProcess(training[,num_features_idx], method=c('knnImpute'))

ptraining <- cbind(training$classe, predict(preModel, training[,num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[,num_features_idx]))
prtesting <- cbind(raw_testing$classe, predict(preModel, raw_testing[,num_features_idx]))

names(ptraining)[1] <- 'classe'
names(ptesting)[1] <- 'classe'

#ksvm_model  <- ksvm(classe ~ ., ptraining)
ptraining[is.na(ptraining)] <- 0
ptesting[is.na(ptesting)] <- 0
prtesting[is.na(prtesting)] <- 0

rf_model  <- randomForest(classe ~ ., ptraining)
#gbm_model  <- gbm(classe ~ ., data=ptraining)
ksvm_model  <- ksvm(classe ~ ., data=ptraining)
#rpart_model  <- train(classe ~ ., ptraining, method='rf')

#rf_model  <- train(classe ~ ., training_sub, method='rf')
#lda_model  <- train(classe ~ ., training_sub, method='lda')
#gbm_model  <- train(classe ~ ., training_sub, method='gbm')
#lasso_model  <- train(classe ~ ., training_sub, method='lasso')

testing_pred <- predict(rf_model, ptesting) 
print(table(testing_pred, ptesting$classe))
print(mean(testing_pred == ptesting$classe))

testing_pred <- predict(rf_model, ptraining) 
print(table(testing_pred, ptraining$classe))
print(mean(testing_pred == ptraining$classe))

answers <- predict(rf_model, prtesting) 

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(as.character(answers))
