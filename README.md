# Sentiment Analysis & Machine Learning Fundamentals 

##  Project Overview
This repository documents a comprehensive learning journey in Data Science and Natural Language Processing (NLP). The project focuses on building a **Sentiment Analysis** system to classify IMDB movie reviews as Positive or Negative.

The work demonstrates a progression from foundational dat amanipulation to classical Machine Learning, culminating in an introduction to Deep Learning with Transformers.

##  Repository Structure
* **`Sentiment_Analysis .ipynb`**: The capstone project implementing the full NLP pipeline from data cleaning to Model training (Logistic Regression & DistilBERT).

##  Tech Stack & Learnings
This project was built to master the following technologies:

### 1. Data Manipulation & Computation
* **Pandas:** Manual DataFrame construction, data cleaning, and ETL processes.
* **NumPy:** Matrix operations, arraybroadcasting, and dimensionality handling.

### 2. Natural Language Processing (NLP)
* **Text Preprocessing:**
    * **Regex (`re`):** Removing HTML tags, URLs, and punctuation.
    * **NLTK:** Stopword removal and Tokenization.
    * **Emoji:** Handling non-textual sentiment indicators.
* **Feature engineering:** Implementing **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize text data.

### 3. Machine Learning (Scikit-Lear)
* **Models:** Implementation of **Logistic Regression** for binary classification.
* **Evaluation:** Usage of Accuracy Score and Confusion Matrices.
* **Pipeline:** Splitting data (`train_test_split`) and hyperparameter tuning.

### 4. Advanced Architectures (Brief Introduction)
* **Transformers:** Introduction to **BERT** and **distilBERT** architctures.
* **Hugging Face:** Utilizing pre-trained tokenizers and models for state-of-the-art NLP.

* #IMP:- the pipeline is exported as a pkl file that code is not included in the notebook 

##  Installation
To replicate this environmen, install the following dependencies:

