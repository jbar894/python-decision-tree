import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics  

dataset = pd.read_csv('./SRData.csv')  

dataset.describe()  

X = dataset.drop('Placed', axis=1)  
y = dataset['Placed'] 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  

passrate = 0.2
#An array used to convert the predicted success into binary
PS = []

#An array used to store which prections were correct or not
Correct = []
n = 0

#A count of correctness
CC = 0


for index, row in df.iterrows():
    if row[1] >= passrate:
        PS.append(1)
    else:
        PS.append(0)
    
    if row[0] == PS[n]:
        Correct.append(1)
        CC+=1
    else:
        Correct.append(0)
    n+=1

print("Accuracy", CC/n)
print("Predicted Score", PS)
print("Correct", Correct)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

#print(df.sort_index(0))
#print(df.at[55,'Predicted'])
#print(row[0])