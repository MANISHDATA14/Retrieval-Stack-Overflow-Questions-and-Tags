# Section 1: Data Loading and Cleaning

import pandas as pd

# Load the dataset from CSV file
df = pd.read_csv('stacksample/Questions.csv', encoding='ISO-8859-1', usecols=['Id', 'Title', 'Body'])

# Remove HTML tags from the questions' titles and bodies
df['Title'] = df['Title'].str.replace('<[^<]+?>', '')
df['Body'] = df['Body'].str.replace('<[^<]+?>', '')

# Combine the titles and bodies into a single text column
df['Text'] = df['Title'] + ' ' + df['Body']

# Drop the original Title and Body columns
df = df.drop(['Title', 'Body'], axis=1)

# Load the tags dataset and merge it with the questions dataset based on question ID
tags = pd.read_csv('stacksample/Tags.csv')
df = pd.merge(df, tags, left_on='Id', right_on='Id')

# Limit the number of tags to the top 10 most common ones
tag_counts = df['Tag'].value_counts().head(10).index.tolist()
df = df[df['Tag'].isin(tag_counts)]

# Section 2: Data Preprocessing

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define the preprocessing function
def preprocess(text):
    # Remove non-alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Join the tokens back into a string
    text = ' '.join(tokens)

    return text

# Apply the preprocessing function to the Text column
df['Text'] = df['Text'].apply(preprocess)

# Section 3: Train-Test Split

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Tag'], test_size=0.2, random_state=42)

# Section 4: Feature Extraction

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer object
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer to the training set and transform both the training and testing sets
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Section 5: Model Training and Evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a logistic regression classifier
classifier = LogisticRegression(max_iter=10000)

# Train the classifier on the training set
classifier.fit(X_train, y_train)

# Predict the tags for the testing set
y_pred = classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
