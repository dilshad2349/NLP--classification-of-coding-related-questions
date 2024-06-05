This repository contains the code and resources for training a machine learning model to automatically tag coding-related questions with relevant tags.

Overview
The goal of this project is to develop a model that can classify coding-related questions into predefined tags or categories. This can be useful for organizing and categorizing questions on coding forums, Q&A websites, or support platforms.

Dataset
We used a dataset consisting of coding-related questions collected from GitHub.The dataset includes questions along with their corresponding tags or labels.

Model Architecture
We employed a Support Vector Machine (SVM) classifier for this task. The model uses a TF-IDF vectorizer to convert text data into numerical features and then trains a linear SVM classifier on these features.

Model Training
The training process involves the following steps:

Data Preprocessing: The text data is preprocessed to remove noise, perform tokenization, remove stopwords, and perform lemmatization.

Feature Extraction: Text data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique.

Model Training: The preprocessed and vectorized data is used to train a linear SVM classifier.

Model Evaluation: The trained model is evaluated on a separate validation set to assess its performance in terms of accuracy and other relevant metrics.
