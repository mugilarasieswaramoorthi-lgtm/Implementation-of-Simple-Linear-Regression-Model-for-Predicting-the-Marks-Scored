# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start and Import Libraries

2.Load and Prepare the Dataset

3.Split the Dataset into Training and Testing Sets

4.Train the Linear Regression Model and Make Predictions

5.Evaluate Model Performance and Visualize Results


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mugilarasi E
RegisterNumber: 25017644


import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.metrics import mean_absolute_error,mean_squared_error
 df=pd.read_csv('/student_scores.csv')
 df.head()
 df.tail()
 X=df.iloc[:,:-1].values
 print(*X)
 Y=df.iloc[:,1].values
 print(*Y)
 from sklearn.model_selection import train_test_split
 X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_sta
 from sklearn.linear_model import LinearRegression
 regressor=LinearRegression()
 regressor.fit(X_train,Y_train)
 Y_pred=regressor.predict(X_test)
 #Y_pred
 print(*Y_pred)
 #Y_test
 print(*Y_test)
 plt.scatter(X_train,Y_train,color="orange")
 plt.plot(X_train,regressor.predict(X_train),color="red")
 plt.title("Hours vs Scores(Training Set)")
 plt.xlabel("Hours")
 plt.ylabel("Scores")
 plt.show()
 plt.scatter(X_test, Y_test, color="blue")
 plt.plot(X_test, regressor.predict(X_test), color="green")
 plt.title('Testing set (Hours vs Scores)')
 plt.xlabel("Hours")
 plt.ylabel("Scores")
 plt.show()
 mae = mean_absolute_error(Y_test, Y_pred)
 mse = mean_squared_error(Y_test, Y_pred)
 rmse = np.sqrt(mse)
 print("Mean Absolute Error:", mae)
 print("Mean Squared Error:", mse)
 print("Root Mean Squared Error:", rmse

```

## Output:
<img width="583" height="646" alt="Screenshot 2025-10-06 200809" src="https://github.com/user-attachments/assets/9295509b-f0cc-4627-98a6-9f32cd58fe39" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
