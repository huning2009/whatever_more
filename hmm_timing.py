# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:22:19 2018

@author: lenovo
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
from sklearn import preprocessing # To center and standardize the data.


datalist=[]
close = pd.read_csv('../DailySim/data/simData/closeM.csv',index_col=0,usecols=[0,3])
close.columns=['close']
datalist.append(close)

high = pd.read_csv('../DailySim/data/simData/highM.csv',index_col=0,usecols=[0,3])
high.columns=['high']
datalist.append(high)

low = pd.read_csv('../DailySim/data/simData/lowM.csv',index_col=0,usecols=[0,3])
low.columns=['low']
datalist.append(low)

volume = pd.read_csv('../DailySim/data/simData/volumeM.csv',index_col=0,usecols=[0,3])
volume.columns=['volume']
datalist.append(volume)

data = pd.concat(datalist,axis=1)

return_day = (np.array(close[1:])/np.array(close[:-1]))[4:]-1
logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
logVol = np.log(np.array(volume[5:])+1) - np.log(np.array(volume[:-5]+1))
diffreturn = (np.log(np.array(high))-np.log(np.array(low)))[5:]+0.0001

logreturn_scared = preprocessing.scale(logreturn, axis=0, with_mean=True, with_std=True, copy=False)
logreturn5_scared = preprocessing.scale(logreturn5, axis=0, with_mean=True, with_std=True, copy=False)
diffreturn_scared = preprocessing.scale(diffreturn, axis=0, with_mean=True, with_std=True, copy=False)
logVol_scared = preprocessing.scale(logVol, axis=0, with_mean=True, with_std=True, copy=False)

X = np.column_stack([logreturn_scared,diffreturn_scared,logreturn5_scared,logVol_scared])
datelist = close.index.values.astype(int)[5:]
ts=pd.to_datetime(datelist.astype(str)).values

def nav_plot(latent_states_sequence,data,title):
    plt.figure(figsize=(15,4))
    for i in range(hmm.n_components):
        state = (latent_states_sequence == i)
        #idx = np.append(0,state[:-1])
        data['state %d_return'%i] = data.logreturn.multiply(state.T.tolist()[0],axis = 0) 
        plt.subplot(121)
        plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'latent_state %d'%i)
    plt.title(title)
    plt.legend()
    plt.grid(1)
    
def state_plot(close,ts,iDate,Flag):
    if Flag == 'insample':
        closeidx = np.array(close.values[iDate-995:iDate+5].T.tolist()[0])
        ts_s = ts[iDate-999:iDate+1]
    else:
        closeidx = close.values[iDate+5:iDate+255]
        ts_s = ts[iDate+1:iDate+251]
    sns.set_style('white')
    for i in range(hmm.n_components):
        state = (latent_states_sequence == i).T.tolist()[0]
        plt.subplot(122)
        plt.plot(ts_s[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
    plt.legend()
    plt.title(title)
    plt.grid(1)

def df_aq(ts,logreturn,latent_states_sequence,iDate,Flag):
    if Flag == 'insample':
        data = pd.DataFrame(np.column_stack([return_day[iDate-1000:iDate],latent_states_sequence]),index = ts[iDate-999:iDate+1],columns=['logreturn','state'])
    else:
        data = pd.DataFrame(np.column_stack([return_day[iDate:iDate+250],latent_states_sequence]),index = ts[iDate+1:iDate+251],columns=['logreturn','state'])
    return data
    
name = locals()
nDate=len(data)-250
i=1
for iDate in range(1000,nDate,250):
    X_train = X[iDate-1000:iDate]
    hmm = GaussianHMM(n_components = 2, covariance_type='full', n_iter=20000,tol=0.001).fit(X_train)
    
    X_test = X[iDate-1000:iDate+250]
    latent_states = hmm.predict(X_test)
    
    latent_states_sequence = np.mat(np.ndarray(1000)).T
    latent_states_sequence = np.mat(latent_states[:1000]).T
    title = str(datelist[iDate-999]) +'_' + str(datelist[iDate+1]) + '_insample'
    data_pd = df_aq(ts,logreturn,latent_states_sequence,iDate,'insample')
    nav_plot(latent_states_sequence,data_pd,title)
    state_plot(close,ts,iDate,'insample')
    
    latent_states_sequence = np.mat(np.ndarray(250)).T
    latent_states_sequence = np.mat(latent_states[1000:]).T
    title = str(datelist[iDate+1]) +'_' + str(datelist[iDate+251]) + '_outsample'
    data_pd = df_aq(ts,logreturn,latent_states_sequence,iDate,'outsample')
    nav_plot(latent_states_sequence,data_pd,title)
    state_plot(close,ts,iDate,'outsample')

    name['model_'+str(i)] = pd.concat([data_pd.iloc[:,1], data_pd.iloc[:,1]])

    print ('Model',i,'Down')
    i+=1



