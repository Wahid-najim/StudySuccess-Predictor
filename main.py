# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

# Custom dataset of study hours and scores
data = {
    'Hours': [1, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10],
    'Scores': [20, 25, 30, 35, 40, 42, 48, 50, 60, 70, 80, 85, 90]
}
df = pd.DataFrame(data)

# Data visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Hours', y='Scores')
plt.title('Study Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()

# Preparing data for training
X = df[['Hours']]
y = df['Scores']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Creating a simple GUI using Tkinter
def predict_score():
    try:
        hours = float(entry.get())
        pred_score = model.predict([[hours]])[0]
        messagebox.showinfo("Predicted Score", f"Predicted Score: {pred_score:.2f} out of 100")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number for study hours.")

root = tk.Tk()
root.title("Score Predictor")

label = tk.Label(root, text="Enter study hours:")
label.pack()

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Predict Score", command=predict_score)
button.pack()

root.mainloop()
