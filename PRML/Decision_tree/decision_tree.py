# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:39:38 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd

file = r"data.xlsx"
df = pd.read_excel(file,header =0,index = list(range(1,15)))
df.index = pd.RangeIndex(start=1, stop=15, step=1)

ent = lambda x: - x*np.log2(x)-(1-x)*np.log2(1-x) if x!=0 and x!=1 else 0.


#ID3
def choose_feature(df,chosen_col=None,status=None,annotation = None,a=False):
    
    if chosen_col is not None and status is not None:
        df = df[df[chosen_col]==status].copy()
        df.drop(chosen_col,axis=1,inplace = True)
    else:
        df = df.copy()
    
    i_root = ent(sum(df["PlayTennis"]=="Y")/len(df["PlayTennis"]))
    Gains = []
    for col in df.columns[:-1]:
        i = 0.
        for status_ in df[col].unique():
            P_status = sum(df[col]==status_) / len(df["PlayTennis"])
            P_posi = sum(df[df["PlayTennis"]=="Y"][col]==status_)/sum(df[col]==status_)
            i += P_status*ent(P_posi)
        Gains.append(i_root - i)
    
    k = np.argmax(Gains)

    annotation = "{} {}\n{} ".format(annotation,status,df.columns[k])
    if a:
        print("***************")
        print(annotation)
        print("***************")
        print(df)

    return df,df.columns[k],annotation


#ID3,一步步的往下推，直到subdf全为Yes或No
#subdf,chosen_col,annotation = choose_feature(df)
#subdf,chosen_col,annotation = choose_feature(subdf,chosen_col,"O",annotation,a=1)



def CART(df,show_GINI = False):
    n_feature = len(df.columns[:-1])

    GINI = pd.Series([0.]*n_feature, df.columns[:-1],dtype=np.float)
    best_status = pd.Series([""]*n_feature, df.columns[:-1],dtype=str)
    for col in df.columns[:-1]:
        
        GINIs = []
        for status in df[col].unique():
    
            P_status = sum(df[col]==status) / len(df["PlayTennis"])
            P_posi = sum(df[df["PlayTennis"]=="Y"][col]==status)/sum(df[col]==status)
            
            G = 2*P_posi*(1-P_posi)*P_status
            GINIs.append(G)
            GINI[col] += G
        
        best_status[col] = (df[col].unique()[np.argmin(GINIs)])

    if show_GINI:
        print(GINI,'\n')
        print(best_status,'\n')
    
    feature = GINI.index[GINI.values.argmin()]
    print(feature,best_status[feature])
    
    return GINI.index[GINI.values.argmin()],best_status[GINI.index[GINI.values.argmin()]]


feature,status = CART(df)

subdf0 = df[df[feature]!=status].copy()


#----------------------------------------------
feature,status = CART(subdf0)
subdf1 = subdf0[subdf0[feature]==status].copy()

feature,status = CART(subdf1)
subdf2 = subdf1[subdf1[feature]!=status].copy()


feature,status = CART(subdf2,1)
#subdf3 = subdf2[subdf2[feature]==status].copy()
#feature,status = CART(subdf1)
#subdf4 = subdf3[subdf3[feature]==status].copy()




