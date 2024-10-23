

# SMS Spam Detection

This project implements a spam detection system for SMS messages using machine learning techniques. The goal is to classify messages as either 'spam' or 'ham' (not spam) based on their content. The project includes data preprocessing, exploratory data analysis (EDA), feature extraction, model training, and evaluation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [References](#references)

## Project Overview

This project aims to build a spam detection system using a dataset of SMS messages. The system utilizes various machine learning algorithms, including Random Forest, Logistic Regression, and Support Vector Machine (SVM), to classify messages. The project also employs ensemble methods to improve prediction accuracy.

## Dataset

The dataset used in this project is the **SMS Spam Collection Dataset**, which consists of SMS messages labeled as 'spam' or 'ham'. Each message is accompanied by a label indicating whether it is spam (1) or ham (0).

- **Columns:**
  - `label`: The class label (0 for ham, 1 for spam).
  - `sms`: The content of the SMS message.

## Requirements

Make sure you have the following libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn nltk imbalanced-learn wordcloud
```

## Data Preprocessing

The preprocessing steps include:

1. **Loading the Data:**
   Load the dataset using Pandas.

   ```python
   import pandas as pd
   df = pd.read_csv("SMS Spam Dataset.csv")
   ```

2. **Data Inspection:**
   Check for missing values, basic statistics, and data types.

   ```python
   df.info()
   df.describe()
   df.isnull().sum()
   ```

3. **Cleaning the Text:**
   The `clean_text` function is defined to preprocess the SMS messages. This involves:
   - Lowercasing the text
   - Removing special characters and digits
   - Removing links
   - Tokenization
   - Removing stop words
   - Stemming the words

   ```python
   import re
   from nltk.corpus import stopwords
   from nltk.stem import PorterStemmer
   from nltk.tokenize import word_tokenize

   def clean_text(text):
       text = text.lower()
       text = re.sub(r'[^a-zA-Z\s]', '', text)
       text = re.sub(r'http\S+', '', text)
       words = word_tokenize(text)
       stop_words = set(stopwords.words('english'))
       filtered_words = [word for word in words if word not in stop_words]
       stemmer = PorterStemmer()
       stemmed_words = [stemmer.stem(word) for word in filtered_words]
       return ' '.join(stemmed_words)

   df['clean_text'] = df['sms'].apply(clean_text)
   ```

## Exploratory Data Analysis (EDA)

Several visualizations and analyses are performed to understand the data better:

1. **Top N-Grams Analysis:**
   Bi-grams (two-word combinations) are extracted and visualized to identify common phrases.

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   import matplotlib.pyplot as plt
   import seaborn as sns

   cv = CountVectorizer(ngram_range=(2, 2), stop_words='english')
   X = cv.fit_transform(df['sms'])
   sum_words = X.sum(axis=0)
   words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
   words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]
   ```

   ![Top N-Grams](top_ngrams.png)

2. **Class Distribution:**
   The distribution of spam and ham messages is visualized.

   ```python
   sns.countplot(data=df, x='label')
   ```

   ![Class Distribution](class_distribution.png)

3. **Message Length Distribution:**
   The distribution of message lengths is plotted.

   ```python
   df['message_length'] = df['sms'].apply(len)
   sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
   ```

   ![Message Length Distribution](message_length_distribution.png)

4. **Word Clouds:**
   Word clouds are generated for spam and ham messages to visualize common words.

   ```python
   from wordcloud import WordCloud

   spam_messages = df[df['label'] == 1]['sms'].str.lower()
   ham_messages = df[df['label'] == 0]['sms'].str.lower()

   wordcloud_spam = WordCloud().generate(' '.join(spam_messages))
   wordcloud_ham = WordCloud().generate(' '.join(ham_messages))
   ```

## Model Training

### Balancing the Dataset

To handle class imbalance, Random OverSampling is applied.

```python
from imblearn.over_sampling import RandomOverSampler

X = df.drop('label', axis=1)
y = df['label']

oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)
```

### Splitting the Data

The data is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
```

### Feature Extraction

TF-IDF vectorization is applied to convert text data into numerical format.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

### Training Models

The following models are trained:

1. **Random Forest Classifier**

```python
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
```

2. **Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state=42)
lr_classifier.fit(X_train_tfidf, y_train)
```

3. **Support Vector Machine (SVM)**

```python
from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)
```

4. **Ensemble Classifier**

An ensemble classifier combining all three models is created using hard voting.

```python
from sklearn.ensemble import VotingClassifier

ensemble_classifier = VotingClassifier(estimators=[
    ('random_forest', rf_classifier),
    ('logistic_regression', lr_classifier),
    ('svm', svm_classifier)
], voting='hard')

ensemble_classifier.fit(X_train_tfidf, y_train)
```

## Model Evaluation

The models are evaluated using confusion matrices and classification reports.

```python
from sklearn.metrics import confusion_matrix, classification_report

# Evaluate Random Forest
y_pred_rf = rf_classifier.predict(X_test_tfidf)
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Evaluate Logistic Regression
y_pred_lr = lr_classifier.predict(X_test_tfidf)
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Evaluate SVM
y_pred_svm = svm_classifier.predict(X_test_tfidf)
print("SVM - Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Evaluate Ensemble Classifier
y_pred_ensemble = ensemble_classifier.predict(X_test_tfidf)
print("Ensemble Classifier - Classification Report:")
print(classification_report(y_test, y_pred_ensemble))
```

### Accuracy

The accuracy of each model is reported in the classification report. Here are the typical metrics to look for:

- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall.
- **Support**: The number of actual occurrences of the class in the specified dataset.

### Example Accuracy Report

For illustration, the accuracy for Random Forest might look like this:

```
Random Forest - Classification Report:
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       965
           1       1.00      0.91      0.95       151

    accuracy                           0.98      1116
   macro avg       0.99      0.95      0.97      1116
weighted avg       0.98      0.98      0.98      1116
```

## Usage

To predict whether a new SMS is spam or not, use the `predict_fake_or_real` function:

```python
def predict_fake_or_real(text):
    cleaned_text = clean_text(text)
    text_tfidf = tfidf

_vectorizer.transform([cleaned_text])
    prediction = ensemble_classifier.predict(text_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"
```

You can call this function with any SMS text to get a prediction:

```python
new_sms = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize!"
result = predict_fake_or_real(new_sms)
print(f"The message is: {result}")
```

## Conclusion

The SMS Spam Detection project demonstrates the application of machine learning techniques in natural language processing to classify SMS messages as spam or ham. The ensemble model achieved high accuracy, indicating its effectiveness in real-world applications.



