###############Problem 1 ####################################
####Regular expressions,Count Vectorizer,Stemming, Lemmatization, POS Tagging

import pandas as pd
my_tweet = pd.read_csv("C:/Users/usach/Desktop/NLP Topic Modelling/Data.csv",usecols=['text'])#loading dataset

import re
#storing regular expressions in variables
HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '_|!|&lt;|&lt;|&amp;|#'

def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text
#using defined function to clean data using re patterns
my_tweet['text'] = my_tweet.text.apply(clean)

#storing text data in a list
tweet = []
for i in my_tweet['text']:
    tweet.append(i)
    
#importing nltk for pre-processing of data
import nltk

#downloadng stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
#storing english stop words in a variable
stop_words = stopwords.words('English')

#removing stop words
tweet = ' '.join([word for word in tweet if word not in stop_words]) 

# Stemming
stemmer = nltk.stem.PorterStemmer()
stemm_tweet = stemmer.stem(tweet)

# Lemmatization
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemma_tweet = lemmatizer.lemmatize(tweet)

# Sentence Tokenization
from nltk.tokenize import sent_tokenize
sent_token_tweet = sent_tokenize(lemma_tweet)

#POS Tagging
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')

for  i in sent_token_tweet:
    word_tweet = nltk.word_tokenize(i)
    tagged_tweet = nltk.pos_tag(word_tweet)
    print(tagged_tweet)


##################################LDA#################################
# Latent Dirichlet Allocation (LDA)
from gensim.parsing.preprocessing import preprocess_string

#preprocessing data
my_tweet = my_tweet.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(my_tweet)
corpus = [dictionary.doc2bow(text) for text in my_tweet]

#building LDA model with 5 topics
NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

#printing topics
ldamodel.print_topics(num_words=5)

#building model to check error (efficiency) of model
from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(my_tweet, dictionary, ldamodel)
        yield coherence

#defining min and max topics
min_topics, max_topics = 15,20

#calling coherence model
coherence_scores = list(get_coherence_values(min_topics, max_topics))

#visualizing coherence scores in a graph
import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]
ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

##################################LSA##############################################
# Latent Semantic Analysis / Latent Semantic Indexing
# pip install gensim
# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
from gensim import corpora 
from gensim.models import LsiModel
from gensim.parsing.preprocessing import preprocess_string

import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '_|!|&lt;|&lt;|&amp;|#'

#defining some functions to clean the data using RE patterns
def cleans(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    x = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def clean(x):
    x = clean_text(x)
    x = clean_numbers(x)
    x= cleans(x)
    return x

#preparing document by converting every terms into list
def prepare_documents(documents):
    print('Preparing documents')
    documents = [clean(document) for document in documents]
    documents = [preprocess_string(doc) for doc in documents]
    return documents

#creating LSA model
def create_lsa_model(documents, dictionary, number_of_topics):
    print(f'Creating LSA Model with {number_of_topics} topics')
    document_terms = [dictionary.doc2bow(doc) for doc in documents]
    return LsiModel(document_terms, num_topics=number_of_topics, id2word = dictionary)

#calling LSA model
def run_lsa_process(documents, number_of_topics=10):
    documents = prepare_documents(documents)
    dictionary = corpora.Dictionary(documents)
    lsa_model = create_lsa_model(documents, dictionary, number_of_topics)
    return documents, dictionary, lsa_model

# data loading 
import pandas as pd
my_tweet = pd.read_csv("C:/Users/usach/Desktop/NLP Topic Modelling/Data.csv",usecols=['text'])

#running LSA model for data
documents, dictionary, model = run_lsa_process(my_tweet['text'], number_of_topics=5)

#printing 5 LSA model topics
model.print_topics()

# Coherence Model
from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        documents, dictionary, model = run_lsa_process(my_tweet['text'], number_of_topics=num_topics)
        coherence = calculate_coherence_score(documents, dictionary, model)
        yield coherence

min_topics, max_topics = 5, 11

#calling coherence model
coherence_scores = list(get_coherence_values(min_topics, max_topics))

## Plotting coherence scores 
import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10,8))
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores by number of Topics')


###########################Text Summarization############################
import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies


def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])

def summarize(text:str, num_sentences=5):
    
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(text) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]

#Reload the data again because we clean the data inside defined functions anyways
import pandas as pd
my_tweet = pd.read_csv("C:/Users/usach/Desktop/NLP Topic Modelling/Data.csv",usecols=['text'])

tweet = ' '.join([word for word in my_tweet['text']])

summarize(tweet)

summarize(tweet, num_sentences=2)

################################Problem 2#####################################

##################################LDA#################################
article = open('C:/Users/usach/Desktop/NLP Topic Modelling/NLP-TM.txt' , encoding = "utf8")
   
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('English')

new_article = ' '.join([word for word in article if word not in stop_words])

# Latent Dirichlet Allocation (LDA)
from nltk.tokenize import sent_tokenize
sent_token_article = sent_tokenize(new_article)
from pandas import DataFrame
sent_token_article = DataFrame(sent_token_article, columns=['text'])

from gensim.parsing.preprocessing import preprocess_string

new_article = sent_token_article.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(new_article)
corpus = [dictionary.doc2bow(text) for text in new_article]

NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)

from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(new_article, dictionary, ldamodel)
        yield coherence


min_topics, max_topics = 10 , 15
coherence_scores = list(get_coherence_values(min_topics, max_topics))

import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

##################################LSA##############################################
# Latent Semantic Analysis / Latent Semantic Indexing
# pip install gensim
# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
from gensim import corpora 
from gensim.models import LsiModel
from gensim.parsing.preprocessing import preprocess_string

import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    x = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def clean(x):
    x = clean_text(x)
    x = clean_numbers(x)
    return x

def prepare_documents(documents):
    print('Preparing documents')
    documents = [clean(document) for document in documents]
    documents = [preprocess_string(doc) for doc in documents]
    return documents

def create_lsa_model(documents, dictionary, number_of_topics):
    print(f'Creating LSA Model with {number_of_topics} topics')
    document_terms = [dictionary.doc2bow(doc) for doc in documents]
    return LsiModel(document_terms, num_topics=number_of_topics, id2word = dictionary)

def run_lsa_process(documents, number_of_topics=10):
    documents = prepare_documents(documents)
    dictionary = corpora.Dictionary(documents)
    lsa_model = create_lsa_model(documents, dictionary, number_of_topics)
    return documents, dictionary, lsa_model

# data loading 
article = open('C:/Users/usach/Desktop/NLP Topic Modelling/NLP-TM.txt' , encoding = "utf8")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

new_article = ' '.join([word for word in article if word not in stop_words])
from nltk.tokenize import sent_tokenize
sent_token_article = sent_tokenize(new_article)
from pandas import DataFrame
sent_token_article = DataFrame(sent_token_article, columns=['text'])

#running LSA model
documents, dictionary, model = run_lsa_process(sent_token_article['text'], number_of_topics=5)

#printing 5 LSA model topics
model.print_topics()

# Coherence Model
from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        documents, dictionary, model = run_lsa_process(sent_token_article['text'], number_of_topics=num_topics)
        coherence = calculate_coherence_score(documents, dictionary, model)
        yield coherence

min_topics, max_topics = 5, 11

coherence_scores = list(get_coherence_values(min_topics, max_topics))
documents

## Plot
import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10,8))
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores by number of Topics')

###########################Text Summarization############################
import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies


def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])

def summarize(text:str, num_sentences=5):
    
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(text) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]

#Reload the fresh data again because we clean the data inside defined functions anyways
article = open('C:/Users/usach/Desktop/NLP Topic Modelling/NLP-TM.txt' , encoding = "utf8")

my_article = ' '.join([word for word in article])

summarize(my_article)

summarize(my_article, num_sentences=2)

