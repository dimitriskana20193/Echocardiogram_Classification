import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tree import DecisionTree as DT 
from forrest import R_Forrest as R_F
from Reads import Read_Handle as R
from helpers import confusion_matrix,accuracy,RMSE
df = R('/home/dimitriskana/workspace/Echocardiogram/echocardiogram.data')
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute
D = DT(min_samples_split=2, max_depth=200)
Imputers = [SoftImpute(),KNN(),NuclearNormMinimization()]
rm = []
accs = [ ]
for item in Imputers : 
    new_df = pd.DataFrame()
    new_df = pd.DataFrame(item.fit_transform(df), columns = df.columns)
    
    new_df = new_df.drop('Still_Alive',axis=1)
    
    #norm
    X = (new_df*new_df.mean()/new_df.std()).to_numpy()
    #target
    y =new_df['Alive_at_1'].to_numpy().astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=1234)
    D.fit(X_train,y_train)
    Pr = D.predict(X_test)
    rm.append(RMSE(Pr,y_test))
    accs.append(accuracy(Pr,y_test))
    print(f'The imputation is done with : {item}')
    print(f'The confusion matrix is: {confusion_matrix(Pr,y_test)}')
print(f'The mean accuracy of the attempts above is :  {np.mean(accs):.4f}')
print(f'The mean rmse of the attempts above is :  {np.mean(rm):.4f}')