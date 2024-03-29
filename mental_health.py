import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class gaussNB:
    """
    This class builds a Gaussian Naive Bayes model fit to the tokenized data.
    """
    def __init__(self) -> None:
        pass
    
    def run(self, dframe):
        """
        This function acts as a driver for the generation of a GNB model.
        Tokenizes text data, generates train/test split, and fits data to model.

        Args:
            dframe (pandas dataframe): dataframe containing all of the text data.
        """
        vectorizer = CountVectorizer(max_features=15)
        X = vectorizer.fit_transform(dframe["text"])
        y = dframe["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = GaussianNB()
        model.fit(X_train.toarray(), y_train)

        y_pred = model.predict(X_test.toarray())
        acc = accuracy_score(y_test, y_pred)
        print("Naive Bayes Accuracy: {:.2f}%".format(acc * 100))
        report = classification_report(y_test, y_pred)
        print(report)
        
        
class logreg:
    """
    This class builds a logistic regression model fit to the data.
    """
    def __init__(self) -> None:
        pass
    
    def run(self, dframe):
        """
        This function acts as a driver for the generation of a logistic regression model.
        Tokenizes text data, generates train/test split, fits data to model, and plots feature importance.

        Args:
            dframe (Dataframe): Pandas Dataframe containing the mental_health information.
        """
        #tokenizing input
        vectorizer = CountVectorizer(max_features=15)
        X = vectorizer.fit_transform(dframe["text"])
        y = dframe["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        #reporting
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Log Reg Accuracy: {:.2f}%".format(acc * 100))
        report = classification_report(y_test, y_pred)
        print(report)
        
        #visualization: feature importance 
        feature_names = vectorizer.get_feature_names_out()
        coeffs = model.coef_[0]
        
        #sort features
        sorted_indices = np.argsort(feature_names)
        sorted_feature_names = [sorted_indices[x] for x in sorted_indices]
        sorted_coeffs = coeffs[sorted_indices]
        print(feature_names)
        
        #display
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_feature_names)), sorted_coeffs, color='skyblue')
        plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
        plt.xlabel('Coefficient (Feature Importance)')
        plt.ylabel('Feature')
        plt.title('Logistic Regression Feature Importance Plot')
        plt.tight_layout()
        plt.show()
        
        #confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
        cm_display.plot(include_values=True, cmap="Blues")
        plt.title("Confusion Mat Log Reg")
        plt.show()
        
        

if __name__ == "__main__":
    df = pd.read_csv("mental_health.csv")
    
    logistic_model = logreg()
    logistic_model.run(df)
    #NB_model = gaussNB()
    #NB_model.run(df)
