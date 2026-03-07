"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value."""

# from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# import random

import math
from collections import defaultdict
import json

trigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)
vocab = set()

sentiment_vectorizer = None
sentiment_model = None

"""
trainFile: a text file, where each line is arbitratry human-generated text
Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
"""
def calcNGrams_train(trainFile):
    with open(trainFile, newline='', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            words = line.split()
            tokens = ["<s>", "<s>"] + words + ["</s>"]
            
            for tok in words:
                vocab.add(tok)
                
            for i in range (2, len(tokens)):
                trigram_counts[(tokens[i-2], tokens[i-1], tokens[i])] += 1
                bigram_counts[tokens[i-2], tokens[i-1]] += 1
    pass #don't return anything from this function!

"""
sentences: A list of single sentences. All but one of these consists of entirely random words.
Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
"""
def calcNGrams_test(sentences):
    vocab_size = len(vocab)
    print(f"Vocab Size: {vocab_size}")
    worst_score = 0
    worst_index = 0
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip(). lower()
        if not sentence:
            continue
        words = sentence.split()
        padded = ['<s>', '<s>'] + words + ['</s>']
        
        score = 0.0
        n = 0

        for i in range(2, len(padded)):
            w1, w2, w3 = padded[i-2], padded[i-1] , padded[i]

            trigram_count = trigram_counts.get((w1, w2, w3), 0)
            bigram_count = bigram_counts.get((w1, w2),0)
            # print("\nSentence:", sentence)
            prob = (trigram_count + 1) / (bigram_count + vocab_size)
            # print((w1, w2, w3), "->", trigram_count, "/", bigram_count)
            score += math.log(prob)
            
            if score < worst_score:
                worst_score = score
                worst_index = idx
        
    return worst_index
"""
trainFile: A jsonlist file, where each line is a json object. Each object contains:
	"review": A string which is the review of a movie
	"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
"""
def calcSentiment_train(trainFile):
    global sentiment_model, sentiment_vectorizer
    obj = []
    labels = []
    reviews = []
    with open (trainFile, "r", encoding = "utf-8") as f:
    	for line in f:
            line = line.strip().lower()
            if not line:
                continue
            words = line.split()
            print(line)
            obj = json.loads(line)
            reviews.append(obj["review"])
            labels.append(obj["sentiment"])
    sentiment_vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    X_train = sentiment_vectorizer.fit_transform(reviews)
    
    sentiment_model = MultinomialNB()
    sentiment_model.fit(X_train, labels)
    pass #don't return anything from this function!

"""
review: A string which is a review of a movie
Return a boolean which is the predicted sentiment of the review.
Must run in under 120 seconds, and must use Naive Bayes
"""
def calcSentiment_test(review):
    X_test = sentiment_vectorizer.transform([review])
    return bool(sentiment_model.predict(X_test)[0])