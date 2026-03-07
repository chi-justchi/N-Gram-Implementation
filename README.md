# N-Gram Implementation and Sentiment Classification

**Group member:** Chi Vo

This project implements two Natural Language Processing techniques:

1. **Trigram Language Modeling**
2. **Naive Bayes Sentiment Classification**

The implementation was completed as part of an NLP assignment.

---

# Project Overview

This project explores two core NLP tasks:

- Building an **n-gram language model** to analyze sentence probability
- Using **Naive Bayes** to classify movie reviews as positive or negative

The goal is to understand how statistical models can capture patterns in natural language.

---

# Part 1: Trigram Language Model

In the first part of the project, we implemented a **trigram language model**.

A trigram model estimates the probability of a word based on the two previous words:

\[
P(w_3 | w_1, w_2)
\]

## Training

The model is trained using a corpus of real human-generated text.

During training we compute:

- **Trigram counts:** `(w1, w2, w3)`
- **Bigram context counts:** `(w1, w2)`

These statistics allow us to compute trigram probabilities.

Sentence padding is used to represent sentence boundaries: `<s> <s> word1 word2 ... wordN </s>`

## Testing

During testing, the program receives a list of sentences:

- Exactly **one sentence is randomly generated**
- The others were generated using an **n-gram model**

The algorithm:

1. Computes the probability of each sentence using the trigram model
2. Uses **log probabilities** to avoid numerical underflow
3. Identifies the sentence with the **lowest probability**

The lowest probability sentence is assumed to be the **random sentence**.

---

# Part 2: Sentiment Classification

The second part of the project implements **sentiment classification using Naive Bayes**.

The training data consists of a **JSON lines file**, where each line contains:

```json
{
  "review": "<movie review text>",
  "sentiment": true
}
```

- **True** → positive review
- **False** → negative review

## Approach

1. Reviews are converted into numerical features using a **bag-of-words representation**.
2. A **Multinomial Naive Bayes classifier** is trained on the review data.
3. The trained model predicts whether a review expresses **positive or negative sentiment**.

The implementation uses the **scikit-learn** library for efficient training.

---

## Technologies Used

- Python
- scikit-learn
- JSON processing
- Natural Language Processing techniques

---

## Files

| File                             | Description                                                   |
| -------------------------------- | ------------------------------------------------------------- |
| `hw2.py`                         | Main implementation of trigram model and sentiment classifier |
| `HW2_grader.py`                  | Provided grader used to evaluate the implementation           |
| `problem1_trainingFile.txt`      | Training data for the trigram model                           |
| `problem2_trainingFile.jsonlist` | Training data for sentiment classification                    |

---

## Key Concepts Demonstrated

- N-gram language models
- Sentence probability estimation
- Log probability scoring
- Naive Bayes classification
- Bag-of-words text representation
