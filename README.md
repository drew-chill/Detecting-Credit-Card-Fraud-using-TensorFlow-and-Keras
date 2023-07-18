# Detecting-Credit-Card-Fraud-using-TensorFlow-and-Keras

Card fraud is a massive source of financial loss for businesses. As there are so many transactions, detecting them manually is an impossible task. We need to rely on automated models to do so.

A model that detects credit card fraud is an example of a classification model. Given some data what is the most likely class (or label) to assign to the data? In the case of credit card fraud, or data is transactions and associated properties (time of day, amount, location, etc.) and the classes are fraudulent or legitimate.

This notebook will walk through how to build a classification model for detecting credit card fraud, by:

1. Obtaining some sample data.
2. Cleaning the sample data.
3. Splitting the data up into training, validation, and test sets.
4. Creating a feed-forward neural network using TensorFlow and Keras, accounting for imbalanced data.
5. Evaluating the model by looking at its ROC curve

## Two very important sources

This document relies heavily on two well-written sources on credit card fraud detection. I refer to both of these frequently and most of the code has been adapted from one of these sources. While they are quite thorough, I aim for this notebook to succinctly create a single model without compromising too much on its performance.

https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook. This book goes into all kinds of detail on what credit card fraud is and creates many different models to detect it. From now on, I'll refer to this as the Handbook.

 Classification of imbalanced data. This is an official TensorFlow tutorial going over building a credit card fraud detection model on a different dataset. From now on, I'll refer to this as the TensorFlow Tutorial.
