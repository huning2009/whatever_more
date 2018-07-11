import pandas as pd
import numpy as np
from sklearn import preprocessing

def maxmin_process(df,industry):
    df = df[df.sum(axis=1) != 0]

    stk = df.columns
    date = df.index

    industry = pd.DataFrame(industry,index=date,columns=stk)

    df_result = pd.DataFrame(np.zeros_like(df),index=date,columns=stk)

    for i in range(1,30):
        df_ind = pd.DataFrame(np.full_like(df,np.nan),index=date,columns=stk)
        df_ind[industry==i]=df
        df_rank = df_ind.rank(axis=1, method='dense', na_option='keep', ascending=True)
        df_rank.fillna(0, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(df_rank.values.T)
        X_train_minmax = pd.DataFrame(X_train_minmax.T,index=date,columns=stk)
        df_result[X_train_minmax!=0]=X_train_minmax

    return  df_result

s1 = pd.read_csv('../DailySim/data/generateScores/lgmWeiRank.csv', index_col=0)
s2 = pd.read_csv('../DailySim/data/generateScores/lgmWeiRank_new.csv', index_col=0)
s3 = pd.read_csv('../DailySim/data/generateScores/lgmWeiRank2.csv', index_col=0)

industry = pd.read_csv('../DailySim/data/simData/industryCiticsM.csv',index_col=0)

s1_new = maxmin_process(s1,industry)
s2_new = maxmin_process(s2,industry)
s3_new = maxmin_process(s3,industry)

scoreCombine = s1_new*0.5 + s2_new*0.25 +s3_new*0.25
scoreCombine = scoreCombine.where(scoreCombine!=0)
scoreCombine.to_csv('../DailySim/data/generateScores/scoreCombine1.csv')