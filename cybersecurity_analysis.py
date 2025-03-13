import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import messagebox, filedialog


class CybersecurityRiskAssessment:
    def __init__(self):
        self.data = None
        self.model = None
        self.encoder = None

    def load_data(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def preprocess_data(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            # Filling missing values
            self.data.fillna(self.data.mean(), inplace=True)
            # Label encoding categorical features
            self.encoder = LabelEncoder()
            for column in self.data.select_dtypes(include=['object']).columns:
                self.data[column] = self.encoder.fit_transform(self.data[column])
            print("Data preprocessed successfully.")
        except Exception as e:
            print(f"Error in preprocessing data: {e}")

    def split_data(self):
        try:
            X = self.data.drop('target', axis=1)  # Assuming 'target' is the target column
            y = self.data['target']
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error splitting data: {e}")
            return None, None

    def train_model(self, X_train, y_train):
        try:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        except Exception as e:
            print(f"Error training model: {e}")

    def evaluate_model(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            print("Classification Report:\n", classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            self.plot_confusion_matrix(cm)
        except Exception as e:
            print(f"Error evaluating model: {e}")

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    def save_model(self, model_path):
        try:
            import joblib
            joblib.dump(self.model, model_path)
            print(f"Model saved at {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, model_path):
        try:
            import joblib
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def interactive_gui(self):
        self.root = tk.Tk()
        self.root.title("Cybersecurity Risk Assessment Tool")
        self.root.geometry('400x300')

        self.label = tk.Label(self.root, text="Select Data File")
        self.label.pack()

        self.load_button = tk.Button(self.root, text="Load Data", command=self.load_data_gui)
        self.load_button.pack(pady=10)

        self.preprocess_button = tk.Button(self.root, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_button.pack(pady=10)

        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_and_evaluate)
        self.train_button.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save Model", command=self.save_model_gui)
        self.save_button.pack(pady=10)

        self.load_button_model = tk.Button(self.root, text="Load Existing Model", command=self.load_model_gui)
        self.load_button_model.pack(pady=10)

        self.root.mainloop()

    def load_data_gui(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("CSV files", "*.csv"), ("JSON files", "*.json")))
        if file_path:
            self.load_data(file_path)
            messagebox.showinfo("Success", "Data loaded successfully.")

    def save_model_gui(self):
        model_path = filedialog.asksaveasfilename(defaultextension=".joblib", title="Save model as", filetypes=(("Joblib files", "*.joblib"),))
        if model_path:
            self.save_model(model_path)
            messagebox.showinfo("Success", "Model saved successfully.")

    def load_model_gui(self):
        model_path = filedialog.askopenfilename(title="Select a model to load", filetypes=(("Joblib files", "*.joblib"),))
        if model_path:
            self.load_model(model_path)
            messagebox.showinfo("Success", "Model loaded successfully.")

    def train_and_evaluate(self):
        try:
            X_train, X_test, y_train, y_test = self.split_data()
            if X_train is not None and y_train is not None:
                self.train_model(X_train, y_train)
                self.evaluate_model(X_test, y_test)
                messagebox.showinfo("Success", "Model trained and evaluated successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = CybersecurityRiskAssessment()
    app.interactive_gui()