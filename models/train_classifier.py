# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath: path to SQLite db
    Output:
        X : feature DataFrame
        Y : label DataFrame
        category_names: used for data visualization (app)
    """

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text: list of text messages (english)
    Output:
        clean_token: tokenized text, clean for ML modeling
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = { 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Arguments:
        model : Scikit ML Pipeline
        X_test : test features
        Y_test : test labels
        category_names : label names (multi-output)
    """
    y_pred=model.predict(X_test)
    labels = np.unique(y_pred)
    i=-1
    for column in Y_test.columns:
        i=i+1
        print (classification_report(Y_test[column], y_pred[:,i], labels= labels))
    pass


def save_model(model, model_filepath):
    """
    Arguments:
        model : Scikit Pipeline object
        model_filepath : destination path to save .pkl file
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
