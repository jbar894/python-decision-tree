import pandas as pd  
import numpy as np  
from sklearn import metrics  
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split  
from sklearn.externals import joblib
import xgboost as xgb

passrate = 0.5
testsize = 0.2

COLUMNS = (
    'CCFR',
    'AMFR',
    'OFR',
    'SFR',
    'PFR',
    'Placed'
)

#read in data
with open('./SRData.csv', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=0 ,names=COLUMNS)

train_features = raw_training_data.drop('Placed', axis=1)

print(raw_training_data)

train_labels = (raw_training_data['Placed'] == 1)

for i in range (0,len(train_labels)):
    if train_labels[i] == True:
        train_labels[i] = 1
    else:
        train_labels[i] = 0

with open('./SRData.csv', 'r') as test_data:
    raw_test_data = pd.read_csv(test_data, header=0, names=COLUMNS)

test_features = raw_test_data.drop('Placed', axis=1)

test_labels = (raw_test_data['Placed'] == 1)

for i in range (0,len(test_labels)):
    if test_labels[i] == True:
        test_labels[i] = 1
    else:
        test_labels[i] = 0


print(test_features)

dtrain = xgb.DMatrix(train_features, train_labels)
dtest = xgb.DMatrix(test_features)

bst = xgb.train({}, dtrain, 20)
bst.save_model('./model.bst')

