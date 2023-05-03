import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class LDA:
    def __init__(self) -> None:
        pass
    
    def run(self, dframe):
        pass


class logreg:
    """
    This class builds a logistic regression model fit to the data.
    """
    def __init__(self) -> None:
        pass
    
    def run(self, dframe):
        """
        This function acts as a driver for the generation of a logistic regressions model.
        Tokenizes text data, generates train/test split, and fits data to model.

        Args:
            dframe (Dataframe): Pandas Dataframe containing the mental_health information.
        """
        vectorizer = CountVectorizer(max_features=15)
        X = vectorizer.fit_transform(dframe["text"])
        y = dframe["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(acc * 100))
        report = classification_report(y_test, y_pred)
        print(report)
        

if __name__ == "__main__":
    df = pd.read_csv("mental_health.csv")
    logistic_model = logreg()
    logistic_model.run(df)
    LDA_model = LDA()
    LDA_model.run(df)
