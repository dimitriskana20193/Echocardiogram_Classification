import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tree import DecisionTree as DT 
from forrest import R_Forrest as R_F
from Reads import Read_Handle as R
from helpers import confusion_matrix,accuracy,RMSE
df = R('/home/dimitriskana/workspace/Echocardiogram/echocardiogram.data')
Drop_Data = df.dropna().drop('Still_Alive',axis=1)
  
X = Drop_Data.drop('Alive_at_1', axis =1)
X = (X*X.mean()/X.std()).to_numpy()
#target
y =Drop_Data['Alive_at_1'].to_numpy().astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y,
 test_size = 0.33, random_state=1234)
D = DT(min_samples_split=2, max_depth=200)
D.fit(X_train,y_train)
CM = confusion_matrix(D.predict(X_test),y_test)
print(f'The confusion matrix of the Decision tree: {CM }')
F = RMSE(D.predict(X_test),y_test)
print(f'The RMSE of the Decision tree model is : {F :.4f}')
print(f'The accuracy of the Decision tree model is : {accuracy(D.predict(X_test),y_test) :.4f}')

Forest = R_F(min_samples_split=2, max_depth=200,n_trees=2)
Forest.fit(X_train,y_train)
XX = Forest.predict(X_test)
print(f'The RMSE of the random forrest : {RMSE(XX,y_test) :.4f}')
print(f'The accuracy of the random forrest is : {accuracy(XX,y_test) :.4f}')

