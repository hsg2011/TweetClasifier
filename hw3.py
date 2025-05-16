#%%
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import sklearn 
import string
import warnings
from sklearn.metrics import accuracy_score
import re # helps you filter urls
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from IPython.display import display, Latex, Markdown
warnings.filterwarnings('ignore')
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = re.sub(r"https?://\S+", "", text)
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")
    text = text.replace("-", " ")
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    tokens = word_tokenize(text)
    
    tokens = [t for t in tokens if t.strip() != ""]
    
    tagged_tokens = pos_tag(tokens)
    
    lemmatizer = WordNetLemmatizer()
    final_tokens = []
    
    # Lemmatize
    for word, tag in tagged_tokens:
        wn_tag = posMapping.get(tag[0].upper(), "n")  # default to noun
        try:
            lemma = lemmatizer.lemmatize(word, wn_tag)
            final_tokens.append(lemma)
        except Exception:
            # If lemmatization fails, skip this token
            continue
    
    return final_tokens
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df_copy = df.copy()   
    df_copy['text'] = df_copy['text'].apply(lambda t: process(t, lemmatizer))
    return df_copy
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,          
        preprocessor=lambda x: x,       
        lowercase=False,               
        stop_words=stop_words,         
        min_df=2                     
    )
    
    features = vectorizer.fit_transform(processed_tweets["text"])
    return vectorizer, features

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    names = {'realDonaldTrump', 'mike_pence', 'GOP'}
    
    labels = processed_tweets['screen_name'].apply(lambda name: 0 if name in names else 1)
    
    return labels.to_numpy()
    
#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.mode_label = None

    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        num = np.bincount(y)
        self.mode_label = np.argmax(num)
        return self

    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        return np.full(shape=(X.shape[0],), fill_value=self.mode_label)

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
    classifier = SVC(kernel=kernel)
    classifier.fit(X_train, y_train)
    
    return classifier

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    predictions = classifier.predict(X_validation)
    accuracy = accuracy_score(y_validation, predictions)
    
    return accuracy

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    processed_texts = unlabeled_tweets["text"].apply(process)
    X_test = tfidf.transform(processed_texts)
    predictions = classifier.predict(X_test)
    
    return predictions