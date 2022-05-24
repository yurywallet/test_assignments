# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:20:39 2018

@author: Yury
"""
import pandas as pd
import numpy as np

#two databases
#c=column name which is to encode
def OHE(train, test, c):
    # frequency of values
    a=train[c].value_counts()
    a=pd.DataFrame(a)
    a.columns=['count']
    a['val']=a.index
    val=list(a.index)
    a['perc']=a['count']/(a['count'].sum())
    b=a.shape[0]
    a=a.sort_values(['count'], ascending=False)
    '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    a=a[a['perc']>0.03] 
    bb=(a.shape[0]==b)*1 #check if shape is the same (1== no values with low freq)
    val=list(a['val'])
    #keep only frequent values 
    val=['OHE_'+c+'_'+str(x) for x in val]
    
    #train dummies
    cols_hot_train=[]
    oh=pd.DataFrame(index=train.index)
    df_m_train=pd.DataFrame(index=train.index)    
    oh=pd.get_dummies(train[c], prefix='OHE_'+c)
    oh.index=df_m_train.index
    cols_hot_train.extend(list(oh.columns))
    df_m_train = pd.concat([df_m_train, oh], axis=1)
    
    #test dummies
    cols_hot_test=[]
    oh=pd.DataFrame(index=test.index)
    df_m_test=pd.DataFrame(index=test.index)
    oh=pd.get_dummies(test[c], prefix='OHE_'+c)
    oh.index=df_m_test.index
    cols_hot_test.extend(list(oh.columns)) 
    df_m_test= pd.concat([df_m_test, oh], axis=1)
    
    #intersection
    colu=[x for x in cols_hot_test if x in cols_hot_train]
    
    #keep only frequent
    colu=[x for x in val if x in colu] #sorted by freq
    
    #dummy trap
    if bb==1 and len(colu)==b:
        colu=colu[:-1] #remove less frequent

    
    train.drop([c], axis=1, inplace=True)
    test.drop([c], axis=1, inplace=True)
    
    #final
    train=pd.concat([train,df_m_train[colu]], axis=1)
    test=pd.concat([test,df_m_test[colu]], axis=1)
    
    return train, test, colu

#one database,
def OHE_single(train, c):
    
    # frequency of values
    a=train[c].value_counts()
    a=pd.DataFrame(a)
    a.columns=['count']
    a['val']=a.index
    val=list(a.index)
    a['perc']=a['count']/(a['count'].sum())
    b=a.shape[0]
    a=a.sort_values(['count'], ascending=False)
    '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    #drop values less than 3%
    a=a[a['perc']>0.03] 
    bb=(a.shape[0]==b)*1 #check if shape is the same (1== no values with low freq)
    val=list(a['val'])
    
    #keep only frequent values 
    val=['OHE_'+c+'_'+str(x) for x in val]
    
    #train dummies
    cols_hot_train=[]
    
    oh=pd.DataFrame(index=train.index)
    #train
    df_m_train=pd.DataFrame(index=train.index)    
    oh=pd.get_dummies(train[c], prefix='OHE_'+c)
    oh.index=df_m_train.index
    cols_hot_train.extend(list(oh.columns))
    df_m_train = pd.concat([df_m_train, oh], axis=1)
    

    
    #keep only frequent
    colu=[x for x in val if x in cols_hot_train] #sorted by freq
    
    #dummy trap
    if bb==1 and len(colu)==b:
        colu=colu[:-1] #remove less frequent

    #drop initial column
    train.drop([c], axis=1, inplace=True)

    
    #final
    train=pd.concat([train,df_m_train[colu]], axis=1)

    
    return train, colu


def OHE_single_cat(train, cat):
    
    new_columns=[]
    for c in cat:
    
        # frequency of values
        a=train[c].value_counts()
        a=pd.DataFrame(a)
        a.columns=['count']
        a['val']=a.index
        val=list(a.index)
        a['perc']=a['count']/(a['count'].sum())
        b=a.shape[0]
        a=a.sort_values(['count'], ascending=False)
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        #drop values less than 3%
        a=a[a['perc']>0.03] 
        bb=(a.shape[0]==b)*1 #check if shape is the same (1== no values with low freq)
        val=list(a['val'])
        
        #keep only frequent values 
        val=['OHE_'+c+'_'+str(x) for x in val]
        
        #train dummies
        cols_hot_train=[]
        
        oh=pd.DataFrame(index=train.index)
        #train
        df_m_train=pd.DataFrame(index=train.index)  
        
        oh=pd.get_dummies(train[c], prefix='OHE_'+c)
        
        oh.index=df_m_train.index
        cols_hot_train.extend(list(oh.columns))
        df_m_train = pd.concat([df_m_train, oh], axis=1)
        
    
        
        #keep only frequent
        colu=[x for x in val if x in cols_hot_train] #sorted by freq
        
        #dummy trap
        if bb==1 and len(colu)==b:
            colu=colu[:-1] #remove less frequent
    
        #drop initial column
        train.drop([c], axis=1, inplace=True)
    
        
        #final
        train=pd.concat([train,df_m_train[colu]], axis=1)
        
        #extend new feature columns
        new_columns.extend(colu)
    
    
    return train, new_columns

def one_hot_encoder_orig(data, nan_as_category = True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis = 1, inplace = True)
    
    return data, [c for c in data.columns if c not in original_columns]

def one_hot_encoder(data, tiny=0.03, nan_as_category = True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
        
        values = list(data[c].unique())
        
        a=data[c].value_counts()
        a=pd.DataFrame(a)
        a.columns=['count']
        a['val']=a.index
        a['perc']=a['count']/(a['count'].sum())
        b=a.shape[0]
        a=a.sort_values(['count'], ascending=False)
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        #drop values less than 3%
        a=a[a['perc']>tiny] 
        
        bb=(a.shape[0]==b)*1 #check if shape is the same (1== no values with low freq)
        
        #list of frequent values
        values=list(a['val'])
        
        #dummy trap
        if bb==1 and len(values)==b:
            values=values[:-1] # remove 1 less frequent
            
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
            
    data.drop(categorical_columns, axis = 1, inplace = True)
    
    return data, [c for c in data.columns if c not in original_columns]