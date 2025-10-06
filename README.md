# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mugilarasi E
RegisterNumber: 25017644


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    'Hours': [1.1, 2.5, 3.2, 4.5, 5.1, 6.7, 8.0, 2.8, 3.5, 1.5, 9.2, 5.5, 3.3, 2.2, 7.7, 4.8, 6.9, 7.4, 3.9, 1.9, 8.5, 2.7, 5.9, 4.3, 9.7],
    'Scores': [22, 45, 50, 54, 60, 65, 78, 48, 52, 30, 88, 66, 55, 40, 85, 62, 75, 80, 58, 38, 82, 50, 72, 54, 95]
}

df = pd.DataFrame(data)
df.to_csv('student_scores.csv', index=False)

X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values    

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("Predicted values:", Y_pred)
print("Actual values:", Y_test)

plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

```

## Output:
<img width="583" height="646" alt="Screenshot 2025-10-06 200809" src="https://github.com/user-attachments/assets/9295509b-f0cc-4627-98a6-9f32cd58fe39" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
