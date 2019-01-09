# SENTIMENT ANALYSIS
 
# Importing the dataset
import pandas as pd
dataset = pd.read_csv('Sentiment_Analysis_Test_Data.tsv', delimiter = '\t')

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1200)  
X_test = cv.fit_transform(corpus).toarray()

# Loading the model
from sklearn.externals import joblib 
classifier_MNBC = joblib.load('MNBC.pkl')  
  
# Using the loaded model to make predictions 
dataset['Prediction'] = classifier_MNBC.predict(X_test) 

