import pandas as pd 
import numpy as np 
""""Epss : Large number -> abnormal
    Lvdd: Large hearts tend to be sick hearts
    Wall_Motion_index: should be around 12-13 """
def Read_Handle(url):
    col_names = ['Months_Survived','Still_Alive','Age_o_HA',
    'Effusion','Fractional_Short','Epss',
    'Lvdd','Wall_MotionSc','Wallmotion_index','Mult',
    'Name','Group','Alive_at_1']
    if url == None :
        raise ValueError(f'Provide valid path to data')
    p = pd.read_csv(url,  on_bad_lines= 'skip', names = col_names)
    Data = p.drop(['Wall_MotionSc','Name','Group','Mult'],axis=1)
    Data[Data =='?' ] = np.nan
    Data = Data.astype(float)

    id = Data[(Data['Months_Survived']<12) & (Data['Alive_at_1']!=1)]
    #clear the nans for patients that survived less than a year, 
    # in the target column I assign the still_alive value
    Data.loc[id.index,'Alive_at_1'] = Data.iloc[id.index]['Still_Alive']
    Data.loc[Data['Months_Survived']>12,'Alive_at_1'] = 1
    #check if patients that are still alive after 12 months,
    #  are labeled as alive
    Data.loc[(Data['Months_Survived']>12)&Data['Still_Alive']==1,'Alive_at_1'] = 1
    # we create a dataframe without patients that are alive less than a year
    df = Data[~((Data['Months_Survived']<12) & (Data['Alive_at_1']==1))].reset_index().drop('index',axis =1)
    #we also want to get rid some more nans for our target,
    #  we assign the value of  the Still_Alive column to our target   
    i = df[df['Alive_at_1'].isna()].index
    df.loc[i,'Alive_at_1'] = df.loc[i,'Still_Alive']
    #droping the still alive as it is incoporated to the alive at 1 and the months_survived
    # we use standardization for the X data only for the non categorical values 
    return df


