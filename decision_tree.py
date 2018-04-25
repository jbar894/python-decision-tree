import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics  

dataset = pd.read_csv('./SRData.csv')  
#print(dataset.describe())  

passrate = 0.5
testsize = 0.2

X = dataset.drop('Placed', axis=1)  
y = dataset['Placed'] 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=0) 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  


#An array used to convert the predicted success into binary
PS = []

#An array used to store which prections were correct or not
Correct = []
n = 0

#A count of correctness
CC = 0

#Converting to a 0 or 1 with a given pass rate then determining if it is correct or not
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

cdf=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred, 'PS': PS}) 

print(cdf)

print("Accuracy", CC/n)
#print("Predicted Score", PS)
#print("Correct", Correct)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

sj = {'AMFR':[0.379310], 'OFR':[0.437500], 'SFR':[0.421875], 'PFR':[0.381119]}
sjdf = pd.DataFrame(data=sj)

sj_pred = regressor.predict(sjdf)
print("Single job prediction", sj_pred)
