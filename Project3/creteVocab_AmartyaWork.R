setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)

# replacement
train$review = gsub('<.*?>', ' ', train$review)


install.packages('text2vec')
library(text2vec)

library(glmnet)


stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "the", "us")

# iterators over input objects in order to create vocabularies
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)

# ngram integer vector. 
# The lower and upper boundary of the range of n-values for different n-grams to be extracted.

tmp.vocab = create_vocabulary(it_train, 
                              stopwords = stop_words, 
                              ngram = c(1L,4L))

# This function filters the input vocabulary and throws out very frequent and very infrequent terms. 
tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                             doc_proportion_max = 0.5,
                             doc_proportion_min = 0.001)

# Convert a character vector to a document term matrix
# the frequency of terms that occur in a collection of documents
dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))

set.seed(6781)

# generalized linear and similar models via penalized maximum likelihood
tmpfit = glmnet(x = dtm_train, 
                y = train$sentiment, 
                alpha = 1,
                family='binomial')

tmpfit$df

myvocab = colnames(dtm_train)[which(tmpfit$beta[, 44] != 0)]

write.csv(myvocab, file = "vocab.csv")


