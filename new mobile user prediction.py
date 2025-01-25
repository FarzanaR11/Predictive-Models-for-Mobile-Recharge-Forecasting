import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data: User behavior (age, frequency of recharge) and their recharge amount
data = pd.DataFrame({
    "user_age": [25, 32, 28, 40, 22],
    "recharge_frequency": [10, 15, 7, 20, 5],
    "monthly_recharge": [500, 700, 300, 1000, 200]
})

# Model: Predict next month's recharge amount
X = data[["user_age", "recharge_frequency"]]
y = data["monthly_recharge"]
model = LinearRegression().fit(X, y)

# Simulate a new user group
new_users = pd.DataFrame({
    "user_age": [30, 27, 35],
    "recharge_frequency": [12, 8, 18]
})

# Predictions
predictions = model.predict(new_users)
print("Predicted Monthly Recharge for New Users (in BDT):", predictions)
