library(glmnet)
library(randomForest)
library("xgboost")

data <- read.csv("Ames_data.csv")

testIDs <- read.table("project1_testIDs.dat")

for (j in 1:2)
{  
train <- data[-testIDs[,j], ]
test <- data[testIDs[,j], ]
test.y <- test[, c(1, 83)]
test <- test[, -83]
write.csv(train,"train.csv",row.names=FALSE)
write.csv(test, "test.csv",row.names=FALSE)
write.csv(test.y,"test_y.csv",row.names=FALSE)
#test.y=read.csv("test_y.csv")

trainData <- read.csv("train.csv")
testdata <- read.csv("test.csv")
## Processing Train Data 
remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')

trainData = trainData[,setdiff(colnames(trainData),remove.var)]

## numerical variable taken using summary function on Amesdata

winsor.vars <- c("Lot_Frontage","Lot_Area","Year_Built", "Year_Remod_Add","Mas_Vnr_Area",       "BsmtFin_SF_1"     
                 ,"BsmtFin_SF_2"       ,"Bsmt_Unf_SF"       , "Total_Bsmt_SF" ,"First_Flr_SF" 
                 ,"Second_Flr_SF"      ,"Gr_Liv_Area"       ,"Bsmt_Full_Bath"     ,"Bsmt_Half_Bath"    
                 ,"Full_Bath"          ,"Half_Bath"          ,"Bedroom_AbvGr"      ,"Kitchen_AbvGr"      ,"TotRms_AbvGrd"     
                 ,"Fireplaces"         ,"Garage_Yr_Blt"      ,"Garage_Cars"        ,"Garage_Area"        ,"Wood_Deck_SF"      
                 ,"Open_Porch_SF"      ,"Enclosed_Porch"     ,"Three_season_porch" ,"Screen_Porch"       ,"Misc_Val"          
                 ,"Mo_Sold"            ,"Year_Sold"    )  

## trainData winsorization 
trainData_win=trainData ## To be used later for test data
quan.value <- 0.95
for(var in winsor.vars){
  tmp1 <- trainData[, var]
  myquan1 <- quantile(tmp1, probs = quan.value, na.rm = TRUE)
  tmp1[tmp1 > myquan1] <- myquan1
  trainData[, var] <- tmp1
}

## calculating Mean from Amesdata and replacig with NULLS
trainData$Garage_Yr_Blt[is.na(trainData$Garage_Yr_Blt)] = 1978

## categoricalImputation  on Train data 
categoricalImputation = function(inputDataframe){
  
  categorical.vars <- colnames(inputDataframe)[
    which(sapply(inputDataframe,
                 function(x) mode(x)=="character"))]
  train.matrix <- inputDataframe[, !colnames(inputDataframe) %in% categorical.vars, 
                                 drop=FALSE]
  n.train <- nrow(train.matrix)
  for(var in categorical.vars){
    mylevels <- sort(unique(inputDataframe[, var]))
    m <- length(mylevels)
    m <- ifelse(m>2, m, 1)
    tmp.train <- matrix(0, n.train, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[inputDataframe[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
  }
  return(train.matrix)
}
trainData = categoricalImputation(trainData)
trainData_cat=trainData
trainData=trainData[,sort(colnames(trainData))]

## Fit both the models 
print('Results of model 1: Ridge Regression model')
trainData<-trainData[sample(nrow(trainData)),]
x = as.matrix(subset(trainData, select=-c(PID,Sale_Price)))
y_train = trainData$Sale_Price
lambdas <- 10^seq(2, -3, by = -.1)
set.seed(0122)
ridge_reg = glmnet(x, log(y_train), nlambda = 20, alpha = 0.7, family = 'gaussian',
                   lambda = lambdas,type.measure='mae')
cv_ridge <- cv.glmnet(x, log(y_train), alpha = 0, lambda = lambdas,nfolds=10)

## XBOOST

set.seed(0822)
xgb.model <- xgboost(x, log(y_train), max_depth = 6,
                     eta = 0.05, nrounds = 5000,
                     subsample = 0.5,
                     verbose = FALSE)

## Preprocess Test data 
testData <- read.csv("C:/MS/psl/KGD/Project1/test.csv")
testData = testData[,setdiff(colnames(testData),remove.var)]

## passing the test value to train data 
for(var in winsor.vars){
  tmp1 <- trainData_win[, var]
  tmp2 <- testData[, var]
  myquan1 <- quantile(tmp1, probs = quan.value, na.rm = TRUE)
  myquan2 <- quantile(tmp2, probs = quan.value, na.rm = TRUE)
  tmp2[tmp2 > myquan2] <- myquan1
  testData[, var] <- tmp2
}

testData$Garage_Yr_Blt[is.na(testData$Garage_Yr_Blt)] = 1978

testData = categoricalImputation(testData)

var1=colnames(trainData_cat)
var2=colnames(testData) 

## Finding the difference in levels between trai nand test

non.matched_1 = var1[!var1 %in% var2]
to_be_added_test=non.matched_1

non.matched_2 = var2[!var2 %in% var1]
to_be_deleted_test=non.matched_2


## Obsolete levels removed from test

testData_tmp = testData[,setdiff(colnames(testData),to_be_deleted_test)]


## Missing levels added from train to test and defaulted to 0

for(i in to_be_added_test){
  testData_tmp[,i] = 0
  
}

## Sales Price is not needed in Train data
testData_tmp = testData_tmp[,setdiff(colnames(testData_tmp),"Sale_Price")]
testData=testData_tmp


## Predicting model
## sorting data
testData=testData[,sort(colnames(testData))]
x_test = as.matrix(subset(testData, select=-c(PID)))
#trainData=trainData[,sort(colnames(trainData))]
optimal_lambda <- cv_ridge$lambda.min
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
predictions_test = exp(predictions_test)

output = data.frame(testData$PID)
output$Sale_Price = predictions_test 
colnames(output) = c("PID","Sale_Price")  
write.csv(output,"mysubmission1.txt",row.names = F,col.names = T)

## Model 2 prediction 
pred <- predict(xgb.model , x_test)
pred_rf = exp(pred)
output2 = data.frame(testData$PID)
output2$Sale_Price = pred_rf
colnames(output2) = c("PID","Sale_Price")  
write.csv(output2,"mysubmission2.txt",row.names = F,col.names = T)


## Evaluation to be deleted before submitting 
pred_1 <- read.csv("mysubmission1.txt")
names(test.y)[2] <- "True_Sale_Price"
pred_1 <- merge(pred_1, test.y, by="PID")
print(sqrt(mean((log(pred_1$Sale_Price) - log(pred_1$True_Sale_Price))^2)))

## Evaluation to be deleted before submitting 
pred_1 <- read.csv("mysubmission2.txt")
names(test.y)[2] <- "True_Sale_Price"
pred_1 <- merge(pred_1, test.y, by="PID")
print(sqrt(mean((log(pred_1$Sale_Price) - log(pred_1$True_Sale_Price))^2)))

}





### Random Forest commented 

#pred_2 <- read.csv("mysubmission2.txt")
#names(test.y)[2] <- "True_Sale_Price"
#pred_2 <- merge(pred_2, test.y, by="PID")
#print(sqrt(mean((log(pred_2$Sale_Price) - log(pred_2$True_Sale_Price))^2)))

#rf <- randomForest(log(Sale_Price)~., data=data.frame(train_rf))
# 
#pred_rf = predict(rf, data.frame(test_rf))
#pred_rf = exp(pred_rf)
# 
#output2 = data.frame(testData$PID)
#output2$Sale_Price = pred_rf
#colnames(output2) = c("PID","Sale_Price")  
