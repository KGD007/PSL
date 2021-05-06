library(pROC)
library(glmnet)

# Read in the vocab 
myvocab = scan(file= "myvocab.txt", what = character())  

set.seed(3402)
train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)

#Remove HTML tags
train$review <- gsub('<.*?>', ' ', train$review)

#Tokenize and create DTM
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

#Run cross validation to find lambda
mylogit.cv = cv.glmnet(x = dtm_train, 
                       y = train$sentiment, 
                       alpha = 0,
                       family='binomial', 
                       type.measure = "auc")
#Fit the model
mylogit.fit = glmnet(x = dtm_train, 
                     y = train$sentiment, 
                     alpha = 0,
                     lambda = mylogit.cv$lambda.min, 
                     family='binomial')
#Predict on test data
test = read.table("test.tsv",
                  stringsAsFactors = FALSE,
                  header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
mypred = predict(mylogit.fit, dtm_test, type = "response")
output = data.frame(id = test$id, prob = as.vector(mypred))

write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')

#Find ROC
test.y = read.table("test_y.tsv", header = TRUE)
pred = read.table("mysubmission.txt", header = TRUE)
pred = merge(pred, test.y, by="id")
roc_obj = roc(pred$sentiment, pred$prob)
tmp = pROC::auc(roc_obj)
print(tmp)