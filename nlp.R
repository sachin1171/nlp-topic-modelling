#Problem Statement-2
#Perform topic modelling and text summarization on the given text data hint use NLP-TM text file.

#--------------Topic Modelling ------------------#

library(tm)
library(slam)
install.packages("topicmodels")
library(topicmodels)

x<-readLines("C:/Users/usach/Desktop/NLP Topic Modelling/NLP-TM.txt")
mydata.corpus <- Corpus(VectorSource(x))

mydata.corpus <- tm_map(mydata.corpus, removePunctuation) 
my_stopwords <- c(stopwords('english'),"brothers", "sisters", "the", "due", "are", "not", "for", "this", "and", "that", "there", "new", "near", "beyond", "time", "from", "been", "both", "than",  "has","now", "until", "all", "use", "two", "ave", "blvd", "east", "between", "end", "have", "avenue", "before", "just", "mac", "being", "when","levels","remaining","based", "still", "off", "over", "only", "north", "past", "twin", "while","then")
mydata.corpus <- tm_map(mydata.corpus, removeWords, my_stopwords)
mydata.corpus <- tm_map(mydata.corpus, removeNumbers)

# build a term-document matrix
mydata.dtm3 <- TermDocumentMatrix(mydata.corpus)
mydata.dtm3

dim(mydata.dtm3)

dtm <- as.DocumentTermMatrix(mydata.dtm3)

rowTotals <- apply(dtm, 1, sum)
dtm.new   <- dtm[rowTotals> 0, ]

library(NLP)
lda <- LDA(dtm.new, 10) # find 10 topics
term <- terms(lda, 5) # first 5 terms of every topic
term

tops <- terms(lda)
tb <- table(names(tops), unlist(tops))
tb <- as.data.frame.matrix(tb)

cls <- hclust(dist(tb), method = 'ward.D2')
par(family = "HiraKakuProN-W3")
plot(cls)

