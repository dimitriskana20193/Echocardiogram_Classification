import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tree import DecisionTree as DT 
from forrest import R_Forrest as R_F
from Reads import Read_Handle as R
from helpers import confusion_matrix,accuracy,RMSE
df = R('/home/dimitriskana/workspace/Echocardiogram/echocardiogram.data')
""""We replace the nans with the median in this segment"""
Mead_Data = df.drop('Still_Alive',axis=1)
Mead_Data.fillna(Mead_Data.median(), inplace=True)
X = (Mead_Data*Mead_Data.mean()/Mead_Data.std()).to_numpy()
#target
y =Mead_Data['Alive_at_1'].to_numpy().astype('int64')


X_train, X_test, y_train, y_test = train_test_split(X, y,
 test_size = 0.33, random_state=1234)
D = DT(min_samples_split=2, max_depth=200)
D.fit(X_train,y_train)
Forest = R_F(min_samples_split=2, max_depth=200,n_trees=2)
Forest.fit(X_train,y_train)
CM = confusion_matrix(D.predict(X_test),y_test)
print(f'The confusion matrix  for the decision tree is : {CM}')
Fm = RMSE(D.predict(X_test),y_test)
print(f'The RMSE for the decision tree is :  {Fm :.4f}')
print(f'The accuracy of the for the decision tree model is : {accuracy(D.predict(X_test),y_test) :.4f}')



print(f'The RMSE for the random forrest is :  {RMSE(Forest.predict(X_test),y_test) :.4f}')
print(f'The accuracy of the for the random forrest model is : {accuracy(Forest.predict(X_test),y_test) :.4f}')