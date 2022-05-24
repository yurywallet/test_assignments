# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:52:08 2019

@author: Yury
"""

'''
#---------------------------------------
def reduce_mem_usage(df, verbose=True):
    return df
#---------------------------------------    
def stat_draw(df,c, cl, ascend=False, he=10, r=2):
    return a, a_extreme, std3, q99
#---------------------------------------   
def woe_distr(df, col, targ, alf=0.5, prnt=True, short=True):
        return df_eda
#---------------------------------------

        
'''


import pandas as pd
from scipy import stats
import numpy as np



def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df

#def memory_reduce(dataset):
#    for col in list(dataset.select_dtypes(include=['int']).columns):
#        if ((np.max(dataset[col]) <= 127) and(np.min(dataset[col]) >= -128)):
#            dataset[col] = dataset[col].astype(np.int8)
#        elif ((np.max(dataset[col]) <= 32767) and(np.min(dataset[col]) >= -32768)):
#            dataset[col] = dataset[col].astype(np.int16)
#        elif ((np.max(dataset[col]) <= 2147483647) and(np.min(dataset[col]) >= -2147483648)):
#            dataset[col] = dataset[col].astype(np.int32)
#    for col in list(dataset.select_dtypes(include=['float']).columns):
#        dataset[col] = dataset[col].astype(np.float32)
#    return dataset

def summary_df(df, short=True):
    print(f"Dataset Shape: {df.shape}")
#    df=df_test

#    t=time.time()
#    sh=df.shape[0]
#    cols=df.columns.to_list()
#    X = pd.DataFrame(df.dtypes,columns=['dtypes'])
#    X = X.reset_index()
#    X.columns = ['col','dtypes']
#    X['Missing'] = df.isnull().sum().values/sh        
#    X['1st_value']=np.nan
#    X['1st_freq']=np.nan 
#    X['2nd_value']=np.nan
#    X['2nd_freq'] =np.nan
#    for c in cols:
#        a=df[c].value_counts().reset_index().head(2)
#        X.loc[X['col']==c,'1st_value']=a.iloc[0,0]
#        X.loc[X['col']==c,'1st_freq']=round(a.iloc[0,1]/sh,4)  
#        if a.shape[0]>1:
#            X.loc[X['col']==c,'2nd_value']=a.iloc[1,0]
#            X.loc[X['col']==c,'2nd_freq'] =round(a.iloc[1,1]/sh,4)   
#    print(time.time()-t)

#    t=time.time()
    if short:
        X=pd.DataFrame(columns=['col','dtype', 'Values','Missing', '1st_value', '1st_freq', '2st_value', '2st_freq'])
    else:
        #entropy - measure of surprice.  in equal probability distrib is the highest
        X=pd.DataFrame(columns=['col','dtype', 'Values','Missing', '1st_value', '1st_freq', '2st_value', '2st_freq', 'Entropy'])
    
    sh=df.shape[0]
    cols=df.columns.to_list()

    for c in cols:   
        a=df[c].value_counts().reset_index()
        # print(c)
        # print(a)
        if a.shape[0]>1:
            ccc=[c,
                             df[c].dtype,
                             a.shape[0],
                             round(df[c].isnull().sum()/sh,4), #df_train.loc[pd.isnull(df_train[c]),c].shape[0]/sh
                             a.iloc[0,0],
                             round(a.iloc[0,1]/sh,4), 
                             a.iloc[1,0],
                             round(a.iloc[1,1]/sh,4)]
            if not short:
                ccc+=[round(stats.entropy(df[c].value_counts(normalize=True), base=2),2)]
            X.loc[X.shape[0]]=ccc
        else :                 
            ccc=[c, df[c].dtype,
                             a.shape[0],
                             round(df[c].isnull().sum()/sh,4), #df_train.loc[pd.isnull(df_train[c]),c].shape[0]/sh
                             a.iloc[0,0],
                             round(a.iloc[0,1]/sh,4), 
                             np.nan,
                             np.nan]
            if not short:
                ccc+=[round(stats.entropy(df[c].value_counts(normalize=True), base=2),2)]
            X.loc[X.shape[0]]=ccc
#    print(time.time()-t)
    
    
    
#    summary['Uniques'] = df.nunique().values
#    summary['First Value'] = df.loc[0].values
#    summary['Second Value'] = df.loc[1].values
#    summary['Third Value'] = df.loc[2].values
#
#    for name in summary['Name'].value_counts().index:
#        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return X


def woe_distr(df, col, targ, alf=0.5, prnt=True, short=True):
#    df=X_tr
#    df=df_tr
#    col=c
#    targ=tar
#    col='id_02'
#    df[col]
#    t=time.time()
    
    sh=df.shape[0]
    #other oprion use groupby
    df_eda=df[[col,targ]].fillna('Missing').groupby(col)[targ].agg(['size','sum']).reset_index()
    df_eda.columns=['Value', 'Rec', 'Bad']
    df_eda=df_eda.sort_values(['Rec'], ascending=[False])
    df_eda['Freq']=df_eda['Rec']/df_eda['Rec'].sum()
    df_eda['BadRate']=df_eda['Bad']/df_eda['Rec']
    df_eda['Value'].fillna('Missing', inplace=True)
#    print(time.time()-t)
    
    
#    t=time.time()    
#    df_eda=pd.DataFrame(columns=['Value', 'Rec', 'Freq', 'Bad', 'BadRate'])
#
#    for v in list(df[col].unique()):
#        if pd.isnull(v):
#            l=df.loc[pd.isnull(df[col]),targ].values
##            v='Missing'
#            df_eda.loc[df_eda.shape[0]]=['Missing',len(l), float(len(l)/sh), float(l.sum()), float(l.sum()/len(l))]
#        else:
#            l=df.loc[df[col]==v,targ].values
#            df_eda.loc[df_eda.shape[0]]=[v,len(l), float(len(l)/sh), float(l.sum()), float(l.sum()/len(l))]
#    print(time.time()-t)
    df_eda['Value']=df_eda['Value'].astype(str)
    df_eda=df_eda.sort_values(['Value'], ascending=[True] )
    df_eda.loc[df_eda.shape[0]]=["Total"]+[df_eda[ccc].sum() for ccc in df_eda.columns.to_list()[1:]]
    df_eda['DB']=(df_eda['Bad']+alf)/(df_eda.loc[df_eda.shape[0]-1,'Bad']+2*alf)
    df_eda['DG']=(df_eda['Rec']-df_eda['Bad']+alf)/(df_eda.loc[df_eda.shape[0]-1,'Rec']-df_eda.loc[df_eda.shape[0]-1,'Bad'] +2*alf)
#    df_eda['WoE']=np.where(df_eda['Bad']==0,0.3,np.log(df_eda['DG'].astype(float)/df_eda['DB'].astype(float)))
#    df_eda['WoE']=df_eda[['DB','DG','BadRate']].apply(lambda x: 0.3 if x['BadRate']==0  else -0.3 if x['BadRate']==1 else (np.log(x['DG']/x['DB'])), axis=1)
    df_eda['WoE']=df_eda[['DB','DG','BadRate']].apply(lambda x: (np.log(x['DG']/x['DB'])), axis=1)

#    df_eda['WoE']=np.where(df_eda['DG']==1,np.log((df_eda['DG']+0.001)/df_eda['DB']),np.log(df_eda['DG']/df_eda['DB']))
    df_eda['IV']=(df_eda['WoE']*(df_eda['DG']-df_eda['DB'])).astype(float).round(4)
    
    df_eda.loc[df_eda.shape[0]-1,'IV']=df_eda['IV'].sum()
    df_eda['BadRate']=(df_eda['Bad']/df_eda['Rec']).astype(float).round(4)
    if prnt==True:
        print("*"*20, col ,"*"*20)
        print(df_eda[['Value', 'Rec', 'Freq', 'Bad', 'BadRate','WoE', 'IV']])
        print(col, "IValue",round(df_eda.loc[df_eda.shape[0]-1,'IV'],4))
    if short:
        return df_eda[['Value', 'Rec', 'Freq', 'Bad', 'BadRate','WoE', 'IV']]
    else:
        return df_eda

def draw_distr(a, c, labels=['Rec',"Bad"],titles=['Value','Number', 'BadRate']):
    import matplotlib.pyplot as plt 
    w=1000
    h=1000
    dpi=150
    
    plt.figure(figsize=(4+w/dpi,4+h/dpi), dpi=dpi)
    # a['Value']=a['Value'].str
    a.sort_index(inplace=True)
    ax1=a[['Value','Rec','Bad']].plot(x='Value',stacked=False, kind="bar")
    ax1.set_xlabel(titles[0])    
    ax1.set_ylabel(titles[1])
    ax2 = ax1.twinx() #share X
    ax2 = a['BadRate'].plot(secondary_y=True, color='tomato', marker='o')
    ax2.set_ylabel(titles[2])
    # or_labels=[x.get_text() for x in ax1.get_xticklabels()]
    # if len(or_labels)!=len(labels): labels=or_labels
    ax1.legend(labels=labels,loc='upper center', bbox_to_anchor=(1, -0.2),
          ncol=5, fancybox=True, shadow=True)
    ax2.legend(labels=[titles[2]], loc='upper center', bbox_to_anchor=(1, -0.35),
          ncol=5, fancybox=True, shadow=True)
    # plt.xticks(rotation=45)
    ax1.xaxis.set_tick_params(rotation=45)
    plt.title(c)
    plt.tight_layout()
    plt.show()



def uni(df, c, labels=['Rec',"Bad"], limit=21):
    a=woe_distr(df,c, 'rev_flag', alf=0.5, prnt=False)
    un_val=df[c].nunique()
    # un_val=a.shape[0]
    if un_val<limit:
        draw_distr(a,c, labels=labels)
    else:
        print(f'Column has {un_val} unique values')
    return a


# c='device'
# a=uni(df,c,labels=['ALL', 'Cancelled'], limit=25)




def high_corr( df, target, threshold = 0.98):
    # Absolute value correlation matrix
    corr_matrix = df[df[target].notnull()].corr().abs()
    
    # Getting the upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print('There are %d columns to remove.' % (len(to_drop)))
    return to_drop

def hard_corr(df, target, threshold = 0.98):
    # Absolute value correlation matrix
    corr_matrix = df.loc[df[target].notnull(),[x for x in df.columns if x!=target]].corr().abs()
    
    # Getting the upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    upper1=(upper>threshold)*1
    upper1=upper1.replace(0,np.nan)
    
    #upper1=upper1[(upper1==1).any(axis=1)]
    upper1.dropna(axis=0, how='all', inplace=True)
    upper1.dropna(axis=1, how='all', inplace=True)
    
    a=list(zip(upper1.index, upper1.columns))
    drop_list=[]
    for i in range(len(a)):
        if abs(df[a[i][0]].corr(df[target]))>abs(df[a[i][1]].corr(df[target])):
            drop_list.append(a[i][1])
        else: drop_list.append(a[i][0])
    drop_list=list(set(drop_list))
    print('There are %d columns to remove.' % (len(drop_list)))
    return drop_list 

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm             import LGBMClassifier
import gc

#Lets check a Covariate Shift of the feature. This means that we will try to distinguish whether a values correspond to a training set or to a testing set.
def covariate_shift(feature, train, test):
#    train=df_train.head(1000)
#    test=df_test.head(300)
#    feature=c
    df_card1_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_card1_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})

    # Creating a single dataframe
    df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
    clf =  LGBMClassifier(n_estimators=500, #silent=True, 
                                     learning_rate=0.1,
                                     metric='auc',
                                     boosting_type='gbdt', #boosting_type='dart',
                                     objective='binary',
                                     nthread=3,
                                     colsample_bytree=1,
                                     max_depth=-1,
                                     random_state=43)
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=47, stratify=df['isTest'])
    df=pd.concat([X_train,y_train], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(df[feature], df['isTest'], test_size=0.15, random_state=41, stratify=df['isTest'])

#    clf = LGBMClassifier(**params, num_boost_round=500)

    model=clf.fit(X_train.values.reshape(-1, 1), y_train,
                eval_set=[ (X_val.values.reshape(-1, 1), y_val)], 
                                        eval_metric= ['auc'], 
                                        verbose= 0, 
                                        early_stopping_rounds= 200)
    roc_auc=model.best_score_['valid_0']['auc']
    #predict that x from train is like from test
    pred_train=clf.predict_proba(df_card1_train[[feature]].values.reshape(-1, 1),num_iteration=clf.best_iteration_)[:,1]
    weigh_like=[-1 if x==1 else x/(1-x) for x in list(pred_train)]
    weigh_like=[max(weigh_like)+1 if x==-1 else x for x in weigh_like]
#    roc_auc =  roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])
#    pd.DataFrame(weigh_like).hist()
#    max(weigh_like)
    del df, X_train, y_train, X_test, y_test, X_val, y_val, model
    gc.collect();
    
    return roc_auc, weigh_like

def stat_draw(df,c, cl, ascend=False, he=10, r=2):
    import matplotlib.pyplot as plt 
    w=500
    h=500
    dpi=100
    plt.figure(
        # figsize=(4+w/dpi,4+h/dpi), 
        dpi=dpi)
    
    a=pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Std','q1%' , 'q5%','q25%', 'Median','q75%', 'q95%', 'q99%'])
    a.loc[a.shape[0]]=[round(df[c].min(),r),round(df[c].max(),r),round(df[c].mean(),r),round(df[c].std(),r), 
                       round(df[c].quantile(0.01),r),round(df[c].quantile(0.05),r),round(df[c].quantile(0.25),r),
                       round(df[c].quantile(0.50),r),round(df[c].quantile(0.75),r),round(df[c].quantile(0.95),r),
                       round(df[c].quantile(0.99),r)]
    

    plt.hist(df[c], density=False, bins=50)
    

    
    
    #add quantiles
    colors = ['blue','pink','red']
    qu=[0.75, 0.95, 0.99]
    for color, q in zip(colors, qu):
        xc=round(df[c].quantile(q),r)
        plt.axvline(x=xc, label=f'q{q} = {xc}', linestyle=':', color=color)
    
    #add deviations
    # colors = ['coral','tomato','firebrick']
    # qu=[1, 2, 3]
    # for color, q in zip(colors, qu):
    #     xc=round(df[c].mean()+q*df[c].std(),r)
    #     plt.axvline(x=xc, label=f'std{q} = {xc}', linestyle='-', color=color)
       # colors = ['coral','tomato','firebrick']
   
    colors = ['tomato','firebrick']
    qu=[2,3]
    for color, q in zip(colors, qu):
        xc=round(df[c].mean()+q*df[c].std(),r)
        plt.axvline(x=xc, label=f'mean+{q}std = {xc}', linestyle='-', color=color)
     
    

    
    tbl=plt.table(cellText=a.values,
              # rowLabels=[' a ', ' b '],
              # rowColours=colors,
               colLabels=a.columns,
               cellLoc = 'center', rowLoc = 'center',
              loc='bottom'     ,
    
               bbox=[-0.4, -0.3, 1.8, 0.2]
              )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    # tbl.set_font_color('darkblue')
    # tbl.scale(2, 2)
    plt.legend()
    plt.title(c)
    plt.tight_layout()
    plt.show()
    
    a_extreme=df[cl].sort_values([c], ascending=ascend).head(he)
    
    std3=round(df[c].mean()+q*df[c].std(),r)
    q99=round(df[c].quantile(0.99),r)
    
    return a, a_extreme, std3, q99