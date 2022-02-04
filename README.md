# Malicious URL detection

It extracts features from malicious URLs and normal URLs, expresses the features as vectors, and uses multiple machine learning to differentiate malicious and normal URLs for features.

## Extract feature based on URL

This project uses multiple machine learning algorithms to detect malicious URLs.Detects/detects malicious URLs when various URLs are input usingThe goal is to develop predictable algorithms. various machines Lexical results for normal and malicious URLs by targeting the learning algorithm. By analyzing the features, the final accuracy of about 96% was derived.

## development environment
- Development Environment: Linux, Jupyter
- Development language: Python3

## Introduction to the applied technology and how to apply it

![image](https://user-images.githubusercontent.com/59435705/152470973-9b1d8d12-1d17-47cf-8e00-5f276b360eaa.png)

 * About 430,000 normal URLs and 150,000 malicious URLs, using a total of 580,000 URL data.
 * 80% of the given URL data is used in the train set to train the model, and 20% is used in the test set to evaluate the model's performance

![image](https://user-images.githubusercontent.com/59435705/152471196-81434be5-c8f8-48cd-8a87-f4fa6bfe3027.png)

 * Based on 22 lexical features extracted from URLs, 8 machine learning algorithms are Create a malicious URL prediction model using

## conclusion
* By analyzing vocabulary characteristics for normal and malicious URLs targeting various machine learning algorithms, about 96% of accuracy was finally derived.
* Compared to the results derived from individual models, the probability of incorrectly predicting malignancy as normal and normal as malicious in multiple models decreases.
* By combining a large number of models, it could be confirmed that higher accuracy was maintained similarly when a specific model was included rather than higher results.

