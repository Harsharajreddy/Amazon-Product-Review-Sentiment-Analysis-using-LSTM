# Amazon-Product-Review-Sentiment-Analysis-using-LSTM

This project aims to analyze Amazon product reviews and predict whether a given review is positive or negative using a Long Short-Term Memory (LSTM) neural network.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction

Sentiment analysis is a powerful tool for understanding customer opinions and feedback. This project utilizes an LSTM network to classify Amazon product reviews as positive or negative based on their content.

## Dataset

The dataset used in this project is `Amazon_Unlocked_Mobile.csv`, which contains reviews of unlocked mobile phones sold on Amazon. Each review is rated on a scale from 0 to 5, where 0 is the most negative and 5 is the most positive.

## Installation

To run this project, you'll need to install the following libraries:

```bash
pip install numpy pandas nltk tensorflow scikit-learn matplotlib
```

## Data Preprocessing

1. Load the dataset and drop unnecessary columns (`Price` and `Review Votes`).
2. Handle missing values in the `Brand Name` column by filling them with 'Notknown'.
3. Convert the ratings into binary sentiments:
   - Ratings >= 3 are considered positive (1)
   - Ratings < 3 are considered negative (0)
4. Preprocess the review texts by:
   - Removing non-alphabetic characters
   - Converting to lowercase
   - Removing stopwords
   - Stemming the words using the Porter Stemmer
5. Tokenize the text data and pad/truncate the sequences to a uniform length.

## Model Architecture

The LSTM model is built using TensorFlow's Keras API. The architecture includes:

- An Embedding layer
- An LSTM layer with 100 units
- Several Dropout layers to prevent overfitting
- Dense layers with ReLU activation
- A final Dense layer with a sigmoid activation for binary classification

## Training the Model

1. Split the data into training and testing sets.
2. Compile the model with `binary_crossentropy` loss and the `adam` optimizer.
3. Train the model for 10 epochs with a batch size of 64.

## Evaluation

Evaluate the model's performance using metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score

Additionally, generate a confusion matrix to visualize the performance.

## Results

The model's performance metrics indicate its effectiveness in classifying the sentiment of Amazon product reviews.

```python
# Example results
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Accuracy: {:.2f}".format(accuracy))
```
