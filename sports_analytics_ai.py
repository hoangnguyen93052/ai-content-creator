import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


class SportsAnalytics:
    def __init__(self, player_data_path, match_data_path):
        self.player_data = pd.read_csv(player_data_path)
        self.match_data = pd.read_csv(match_data_path)

    def preprocess_player_data(self):
        self.player_data['Position'] = self.player_data['Position'].astype('category')
        self.player_data['Position'] = self.player_data['Position'].cat.codes

        self.player_data.fillna(self.player_data.mean(), inplace=True)

        self.features = self.player_data.drop(['Player', 'PerformanceRating'], axis=1)
        self.labels = self.player_data['PerformanceRating']

        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def preprocess_match_data(self):
        self.match_data.fillna(self.match_data.mean(), inplace=True)
        self.match_data['HomeWin'] = (self.match_data['HomeScore'] > self.match_data['AwayScore']).astype(int)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_performance_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        print("Performance Model Accuracy: ", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

    def train_match_outcome_model(self):
        X = self.match_data.drop(['HomeWin'], axis=1)
        y = self.match_data['HomeWin']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.outcome_model = LogisticRegression()
        self.outcome_model.fit(X_train, y_train)
        predictions = self.outcome_model.predict(X_test)

        print("Match Outcome Model Accuracy: ", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

    def predict_player_performance(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        return predictions

    def predict_match_outcome(self, new_match_data):
        predictions = self.outcome_model.predict(new_match_data)
        return predictions

    def save_models(self):
        joblib.dump(self.model, 'performance_model.pkl')
        joblib.dump(self.outcome_model, 'match_outcome_model.pkl')

    def load_models(self):
        self.model = joblib.load('performance_model.pkl')
        self.outcome_model = joblib.load('match_outcome_model.pkl')


if __name__ == "__main__":
    analytics = SportsAnalytics('players.csv', 'matches.csv')
    analytics.preprocess_player_data()
    analytics.preprocess_match_data()
    analytics.train_performance_model()
    analytics.train_match_outcome_model()
    analytics.save_models()

    # Example usage
    new_player_data = np.array([[28, 200, 75, 10]])  # Age, height, weight, etc.
    predicted_performance = analytics.predict_player_performance(new_player_data)
    print("Predicted Performance: ", predicted_performance)

    new_match_data = np.array([[1, 2, 1, 1], [0, 3, 2, 0]])  # HomeScore, AwayScore, etc.
    predicted_outcomes = analytics.predict_match_outcome(new_match_data)
    print("Predicted Match Outcomes: ", predicted_outcomes)