# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:44:14 2019

@author: Yury
"""


import os
kag=0
if kag==0:
    os.environ['MKL_NUM_THREADS'] = '3' #for core i5 5200
    os.environ['OMP_NUM_THREADS'] = '3' #for core i5 5200
else:
    os.environ['MKL_NUM_THREADS'] = '4' #for core i5 5200
    os.environ['OMP_NUM_THREADS'] = '4' #for core i5 5200
    

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns

seed=293423
np.random.seed(293423)
# Set figure width to 12 and height to 9
fig_size=[12,9]

plt.rcParams["figure.figsize"] = fig_size


def plot_stack_hist (dff, col_cats,le):
    if le==1:
        nr=len(col_cats)
        fig, ax = plt.subplots(figsize=(15,60), ncols=3, nrows=nr) 
        rn=0
        for c in col_cats:
        
            table=pd.crosstab(dff[c],dff[tar])
            table=table.div(table.sum(1).astype(float), axis=0)
            
            table.plot(kind='bar', stacked=True, ax=ax[rn][0])
        #    plt.title('Stacked Bar Chart of "'+str(c)+'" vs target', ax=ax[rn][0])
            ax[rn][0].set_title('Stacked Bar Chart of "'+str(c)+'" vs target')
        #    ax[rn][0].xlabel(col_cats)
            plt.xlabel(c)
            plt.ylabel('Proportion of Customers')
            ax[rn][0].set_ylabel('Proportion of Customers', rotation=90, size='small')
        #    ax[rn][0].ylabel('Proportion of Customers')
            
            sns.countplot(x=c, hue=tar, data=dff, ax=ax[rn][1])
            ax[rn][1].set_title('Histogram of "'+str(c)+'" vs target')
            
            sns.boxplot(x=tar, y=c, data=dff, ax=ax[rn][2])
            ax[rn][2].set_title('BOX for "'+str(c)+'" vs target')
                    
            rn+=1 
        #fig.suptitle('Main title')
        fig.tight_layout()
        plt.show()   
    else:
    
        for c in col_cats:
            nr=1
            fig, ax = plt.subplots(figsize=(12,7), nrows=1,  ncols=3, squeeze=False) 
            rn=0
            table=pd.crosstab(dff[c],dff[tar])
            table=table.div(table.sum(1).astype(float), axis=0)
            
            table.plot(ax=ax[rn][0], kind='bar', stacked=True)
        #    plt.title('Stacked Bar Chart of "'+str(c)+'" vs target', ax=ax[rn][0])
            ax[rn][0].set_title('Stacked Bar Chart of "'+str(c)+'" vs target')
        #    ax[rn][0].xlabel(c)
            plt.xlabel(c)
            plt.ylabel('Proportion of Customers')
            ax[rn][0].set_ylabel('Proportion of Customers', rotation=90, size='small')
        #    ax[rn][0].ylabel('Proportion of Customers')
            
            sns.countplot(x=c, hue=tar, data=dff, ax=ax[rn][1])
            ax[rn][1].set_title('Histogram of "'+str(c)+'" vs target')
            
            sns.boxplot(x=tar, y=c, data=dff, ax=ax[rn][2])
            ax[rn][2].set_title('BOX for "'+str(c)+'" vs target')
            
            fig.tight_layout()
            plt.show()
            
            
def plot_hist_singl (dff, col):
#    dff=df
#    col='age'
    nr=1
    fig, ax = plt.subplots(figsize=(12,7), nrows=1,  ncols=3, squeeze=False) 
    rn=0
    table=pd.crosstab(dff[col],dff[tar])
    table=table.div(table.sum(1).astype(float), axis=0)
    
    table.plot(ax=ax[rn][0], kind='bar', stacked=True)
#    plt.title('Stacked Bar Chart of "'+str(c)+'" vs target', ax=ax[rn][0])
    ax[rn][0].set_title('Stacked Bar Chart of "'+str(col)+'" vs target')
#    ax[rn][0].xlabel(c)
    plt.xlabel(col)
    plt.ylabel('Proportion of Customers')
    ax[rn][0].set_ylabel('Proportion of Customers', rotation=90, size='small')
#    ax[rn][0].ylabel('Proportion of Customers')
    
    sns.countplot(x=col, hue=tar, data=dff, ax=ax[rn][1])
    ax[rn][1].set_title('Histogram of "'+str(col)+'" vs target')
    
    sns.boxplot(x=tar, y=col, data=df, ax=ax[rn][2])
    ax[rn][2].set_title('BOX for "'+str(col)+'" vs target')
    
    fig.tight_layout()
    plt.show()

def OHE(dff, df_hot, col, freq, drop_col=False):
    a=pd.DataFrame(dff[col].value_counts()).reset_index()
    a.columns=['val','count']
    a['perc']=a['count']/(a['count'].sum())
    a=a.sort_values(['count'], ascending=False)
    
    #    drop values less than 3%
    #keep only frequent values 
    if sum(a['perc']>freq)==a.shape[0]:
        #dummy trap
        a=a[a['perc']>a['perc'].min()]
    else:
        a=a[a['perc']>freq] 
    val=list(a['val'])

    cols_hot=[]
    
    oh=pd.DataFrame(index=dff.index, dtype=int) 
    oh=pd.get_dummies(dff[col])
    #keep frequent
    oh=oh[val].add_prefix('OHE_'+c+'_')
    
    cols_hot.extend(list(oh.columns))
    if drop_col==True:
        dff.drop([col], axis=1, inplace=True)
    df_hot = pd.concat([df_hot, oh], axis=1)
    
    return df_hot, cols_hot

'''-----------------------------'''

#metric
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

'''-----------------------------'''

'''----------------------------------------------------------'''       
        
''' GET data'''
df=pd.read_excel ('../data/sample scoring data.xlsx', sheet_name='data')


'''target'''
tar='good_bad'
cols=[x for x in df.columns if x!=tar and x!='id code']


di={'bad':1, 'good':0}
df[tar]=df[tar].map(di)

print('BAD clients in df {}'.format(df[tar].sum()/df.shape[0]))


'''-------------------------------------------------------'''

# missing values
#for missing what is the most frequent, mean, other value
df=df.replace([np.inf, -np.inf], np.nan)
df.isna().sum()
#count good/bad for missing
#count good/bad for not-missing grouped
c_miss=[]
for c in cols:
    if df[c].isna().sum()>0:
        c_miss.append(c)

c='job type'
print(df[c].value_counts())

#tmp=df[df[c].isna()]
#print(tmp[c].value_counts())
#print('bad in missing {}'.format(tmp[tar].sum()/tmp.shape[0]))


#df[c+"_m"]=df[c].fillna(df[c].mode().iloc[0])
df[c+"_m"]=df[c].fillna(0)

'''-------------------------------------------------------'''

''' cat or ordinal columns'''
col_cat=[]
col_num=[]

for c in cols:
    a=len(df[c].value_counts())
    if a<=10:
        col_cat.append(c)
    else:
        col_num.append(c)
    print("Column {} has {} different values".format(c, a))
    
for i in range(1,6):
    print("-------------- "+str( i) + " VAlUES-----------------------")
    for c in df.columns:
        a=len(df[c].value_counts())
        if a==i:
            print("Column '{}' has {} different values".format(c, a))
print("-------------------------------------")

'''NEW"'''
'''reorder categories in BAD ascenting order'''


for c in col_cat:
    a=df.groupby(c)[tar].mean().rank(method='dense', ascending=True).astype(int).to_dict()
    df[c]=df[c].apply(lambda x: a[x] if x in a.keys() else x)


''' type'''

df['purpose of loan']=np.where(df['purpose of loan']=='X',10,df['purpose of loan']).astype(int)

'''-------------------------------------------------------'''
'''-------------------------------------------------------'''
#    correlation to target

d={}
print('----Correlation----')
for c in cols:
    if df[c].dtype!="O":
        d[c]=df[tar].corr(df[c])
#        print(c, round(df[tar].corr(df[c]),4))
df_c=pd.DataFrame.from_dict(d, orient='index').reset_index()
df_c.columns=['var', 'corr']
df_c['abs']=np.abs(df_c['corr'])
df_c=df_c.sort_values(by=['abs'], ascending=False)

print(df_c)

df_con=pd.DataFrame()
df_con['col']=col_cat

df_con_n=pd.DataFrame()
df_con_n['col']=col_num

#https://github.com/shakedzy/dython/blob/master/dython/nominal.py

import math
from collections import Counter
import scipy.stats as ss
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

#for c in col_cat:
#    print(c, round(conditional_entropy(df[c], df[tar]),4))

def cramers_v(x,y):
    
    """ 
    Calculates Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:   
        conf_matrix=pd.crosstab(x, y)
        
        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]
        
        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return round(result,6)

dl=pd.DataFrame()
dl['1']=[0,0,0,1,0]
dl['2']=[0,1,0,1,0]
cramers_v(df['job'], df['job type'])

cramers_v(dl['1'], dl['2'])

pd.crosstab(df['job'], df['job type'])

a=[]
for c in col_cat:
    print(c, round(cramers_v(df[c], df[tar]),4))
    a.append(cramers_v(df[c], df[tar]))
df_con['cramer_v']=a
a=[]
for c in col_num:
    print(c, round(cramers_v(df[c], df[tar]),4))
    a.append(cramers_v(df[c], df[tar]))
df_con_n['cramer_v']=a 



def chi_sq(x,y):
    #!!!symmetric coefficient
    confusion_matrix = pd.crosstab(x,y)
#    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
#    chi2 = ss.chi2_contingency(confusion_matrix)
    chi2, p, dof, ex = ss.chi2_contingency(confusion_matrix, correction=False)
#    chi2 The test statistic.
#    p : The p-value of the test
#    dof : Degrees of freedom
#    expected : ndarray, same shape as observed. The expected frequencies, based on the marginal sums of the table
    return chi2, p, dof, ex

a=[]
for c in col_cat:
    print(c, round(chi_sq(  df[c],df[tar])[0],4))
    a.append(chi_sq(  df[c],df[tar])[0])
df_con['chi_2']=a  

a=[]
for c in col_num:
    print(c, round(chi_sq(  df[c],df[tar])[0],4))
    a.append(chi_sq(df[c],df[tar])[0])
df_con_n['chi_2']=a

def theils_u(x, y):
    #!!!Asymmetric!!! coefficient
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    :return: float
        in the range of [0,1]
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


a=[]
for c in col_cat:
    print(c, round(theils_u(df[c], df[tar]),4))
    a.append(theils_u(df[c], df[tar]))
df_con['theil_u']=a  
a=[]
for c in col_num:
    print(c, round(theils_u(df[c], df[tar]),4))
    a.append(theils_u(df[c], df[tar]))
df_con_n['theil_u']=a  








#nr=int(round(len(col_cat)/2,0))
#fig, ax = plt.subplots(figsize=(10,40), ncols=2, nrows=nr)
#rn=0
#cn=0
#for c in col_cat:
##    ax[rn][cn].set_title("Original",                    y = y_title_margin)
#    sns.countplot(x=c, hue=tar, data=df, ax=ax[rn][cn])
#    cn+=1
#    # percentage
#    if cn==2:
#        cn=0
#        rn+=1
#
#plt.show()
#    
#for c in col_cat:
#    table=pd.crosstab(df[c],df[tar])
#    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
#    plt.title('Stacked Bar Chart of "'+str(c)+'" vs target')
#    plt.xlabel(c)
#    plt.ylabel('Proportion of Customers')
#    plt.show()

''' PLOT CATEGORICAL '''
le=0

plot_stack_hist(df, col_cat, le)


plot_stack_hist(df, col_num, le)       
        
    #fig.suptitle('Main title')
#Drop
col_dr=['telephone', 
        'dependents',
        'resident'
        ]
#new features
#'other'

''' LOOK at boxplot if good and bad are separated find the level'''
col_new=[]
c='other'
c1=c+'_bin'
df[c1]=np.where(df[c]==3,1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4),round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4),round(chi_sq(df[c1], df[tar])[0],4))

col_new.append(c1)


c='checking'
c1=c+'_bin'
df[c1]=np.where(df[c]>2,1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4),round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4),round(chi_sq(df[c1], df[tar])[0],4))

col_new.append(c1)


c='job'
c1=c+'_bin'
df['job_bin']=np.where((df['job']==2) | (df['job']==3),1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)


c='housing'
c1=c+'_bin'
df['housing_bin']=np.where((df['housing']==2) | (df['housing']==3),1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)


df['job type_mod']=np.where(df['job type_m']==1,2,df['job type_m'])

c='marital'
c1=c+'_mod'
df['marital_mod']=np.where(df['marital']==4,3,df['marital'])
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

c='marital'
c1=c+'_bin'
df[c1]=np.where((df[c]==4) | (df[c]==3),1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

c='history (number of loans)'
c1=c+'_bin'
df[c1]=np.where(df[c]<1,1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

c='history (number of loans)'
c1=c+'_mod'
df[c1]=np.where(df[c]>1,2,df[c])
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

c='savings'
c1=c+'_bin'
df[c1]=np.where(df[c]>2,0,1)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

c='property'
c1=c+'_bin'
df[c1]=np.where(df[c]==1,0,1)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))
col_new.append(c1)

def cm_qual_bin(c_0,c_1):
    if len(c_0.value_counts())==2 and len(c_1.value_counts())==2:
        cm=confusion_matrix(c_0,c_1)
    #    cm=pd.crosstab(c_0,c_1)
        TN=cm[0,0]
        TP=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        AC=TP+TN
        ER=FN+FP
        s=cm.sum()
        print(cm)
        print("Acc {} ER {} FN {} FP {}".format(AC/s, ER/s, FN/s, FP/s))
    else: print("wrong input")
    
c='principal payments'
c1=c+'_bin'
cram=0.127
p=0
for i in range(0,8000,100):
    df[c1]=np.where(df[c]>4000+i,1,0)
    if cramers_v(df[c1], df[tar])>cram:
        cram=cramers_v(df[c1], df[tar])
        p=4000+i
        print (p, round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))
        
df[c1]=np.where(df[c]>p,1,0)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))

print(df[c1].sum())
cm_qual_bin(df[c1],df[tar])

print(gini_normalized(df[c1],df[tar]))
print(roc_auc_score(df[c1],df[tar]))
col_new.append(c1)

c='age'
c1=c+'_bin'
cram=0
p=0
for i in range(19,70,1):
    df[c1]=np.where(df[c]>i,0,1)
    if cramers_v(df[c1], df[tar])>cram:
        cram=cramers_v(df[c1], df[tar])
        p=i
        print (p, round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))

#p=24
df[c1]=np.where(df[c]>p,0,1)
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))

print(df[c1].sum())
cm_qual_bin(df[c1],df[tar])

print(gini_normalized(df[c1],df[tar]))
print(roc_auc_score(df[c1],df[tar]))
col_new.append(c1)


c='loan duration (m)'
c1=c+'_bin'
cram=0
p=0
for i in range(19,70,1):
    df[c1]=np.where(df[c]>i,0,1)
    if cramers_v(df[c1], df[tar])>cram:
        cram=cramers_v(df[c1], df[tar])
        p=i
        print (p, round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))

df[c1]=np.where(df[c]>p,1,0)
#
print(c, '<>', c1)
print ('Theil ', round(theils_u(df[c], df[tar]),4), '<>',round(theils_u(df[c1], df[tar]),4))
print ('Cramer ', round(cramers_v(df[c], df[tar]),4), '<>',round(cramers_v(df[c1], df[tar]),4))
print ('Chi2 ', round(chi_sq(df[c], df[tar])[0],4), '<>',round(chi_sq(df[c1], df[tar])[0],4))

print(df[c1].sum())
cm_qual_bin(df[c1],df[tar])
print(gini_normalized(df[c1],df[tar]))
print(roc_auc_score(df[c1],df[tar]))
col_new.append(c1)

#c='principal payments'
#c1=c+'_mod'
#df[c1]=np.where(df[c]>6100,3,np.where(df[c]>1100,2,1))
#print (round(cramers_v(df[c], df[tar]),4),round(cramers_v(df[c1], df[tar]),4))
#
#cm=pd.crosstab(df[c1],df[tar])
#print(cm)
#print( 
#      cm.iloc[0,1]/(cm.iloc[0,0]+cm.iloc[0,1]),
#      cm.iloc[1,1]/(cm.iloc[1,0]+cm.iloc[1,1]),
#      cm.iloc[2,1]/(cm.iloc[2,0]+cm.iloc[2,1]))

#col_new=['other_bin', 'job_bin', 'housing_bin', 'job type_mod'
#         ,'marital_mod']

le=0
plot_stack_hist(df, col_new, le)


'''-------------------------------------------------------'''
'''deal with binary variables'''

di1={2:1, 1:0}
di2={1:1, 2:0}
for c in cols:
    if len(df[c].value_counts())==2:
        if(d[c]>0):
           df[c]=df[c].map(di1) 
        else:
           df[c]=df[c].map(di2) 
           
           
'''-------------------------------------------------------'''


def plot_hist(dff, col, bins, integ, max_y,step_y):
#    dff[col].hist(bins=bins)
    mi=min(dff[dff[tar]==1][col].min(),dff[dff[tar]==0][col].min())
    ma=max(dff[dff[tar]==1][col].min(),dff[dff[tar]==0][col].max())
    dff[dff[tar]==0][col].hist(bins=bins, alpha=0.6, range=(mi-1,ma+1), label='0')
    dff[dff[tar]==1][col].hist(bins=bins, alpha=0.6, range=(mi-1,ma+1), label='1')
    plt.title('Histogram of '+col)
    if integ==1:
        plt.xticks(np.arange(dff[col].min(), dff[col].max(), step=int((dff[col].max()-dff[col].min())/10)))
    else:
        plt.xticks(np.arange(dff[col].min(), dff[col].max(), step=round((dff[col].max()-dff[col].min())/10,2) ))
    plt.yticks(np.arange(0, max_y, step=step_y))
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



'''-------------------------------------------------------'''
'''NUMERICAL'''

df['age'].describe()
a=df[df['age']>70]


plot_hist(df, 'age', 30, 1, 80, 5)
plot_hist_singl (df, 'age')

#plt.savefig('hist_age')


df['age_l']=np.log1p(df['age'])
df['age_l'].describe()

c='age_l'
plot_hist(df, 'age_l', 30, 0, 100, 10)

'''---------------------------------------------'''


'''payments'''
c='gross payments'
df[c].describe()

plot_hist(df, c, 30, 1, 250, 5)

a=df[df[c]>12000]

'''------------------'''
c='principal payments'
df[c].describe()


plot_hist(df, c, 30, 1, 150, 5)
plot_hist_singl (df, c)


    
    

#a=df[df[c]>8000]
#
#a=df[df[c]<100]

print('principal vs gross cor=', round(df['principal payments'].corr(df['gross payments']),4))
#df.drop('gross payments', inplace=True, axis=1)

#transform
c='principal payments'
df[c+"_l"]=np.log1p(df[c])
df[c+"_l"].describe()

#c=c+"_l"
plot_hist(df, c+'_l', 30, 0, 100, 10)

c='gross payments'
df[c+"_l"]=np.log1p(df[c])
df[c+"_l"].describe()


#print(chi_sq(  df[c],df[tar])[0],chi_sq(  df[c+'_l'],df[tar])[0])
#print(cramers_v(df[c], df[tar]),cramers_v(df[c+'_l'], df[tar]))
#print(theils_u(  df[c],df[tar]),theils_u(  df[c+'_l'],df[tar]))


'''------------------'''
c='loan duration (m)'
df[c].describe()

plot_hist(df, c, 30, 1, 150, 10)
plot_hist_singl (df, c)

#transform
df[c+"_l"]=np.log1p(df[c])
df[c+"_l"].describe()

plot_hist(df, c+"_l", 30, 0, 100, 10)

'''---------------------------------------------------------'''

tmp=df.groupby(tar).mean()

'''---------------------------------------------------------'''


'''!!!
OHE columns / size trainset...'''
'''-------------------------------------------------------'''

df_ohe=pd.DataFrame(index=df.index, dtype=int) 
col_hot=[]
drop_c=0

for c in col_cat:

#for c in df.columns:
    a=len(df[c].value_counts())
    if a>2 and a<11:
        df_ohe, co=OHE(df,  df_ohe, c, 0.02, drop_c)
        col_hot.extend(co)




#heatmap
#heat=0
#if heat:
#    ax = sns.heatmap(df)    
#    plt.show()


#df.drop(['job type'], inplace=True, axis=1)

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

'''model'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def logg(df, col1, hot, c_hot, folds, spl, see, test_plus_valid):     
    
#    folds=6
#    see=101
#    test_plus_valid=1

    X=df[col1+['id code']]
    
    if hot==1:
        col0=[x for x in c_hot if x not in r]
        X=pd.concat([X,df_ohe[col0]], axis=1)
    y=df[tar]
    C=[0.9,0.8,1.1,1]
    
    
    logreg = LogisticRegression(C=1,
                                solver='lbfgs',
                                multi_class='ovr', 
                                fit_intercept=True,
                                max_iter=1000, 
                                random_state=111)
    

    #test
    av_auc=[]
    av_gini=[]
    av_fn=[]
    av_fp=[]
    
    #validation
    av_auc_v=[]
    av_gini_v=[]
    av_fn_v=[]
    av_fp_v=[]   
    
    #train
    av_train_auc=[]
    av_train_gini=[]
    av_train_fn=[]
    av_train_fp=[] 
    
#    spl=0.15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=spl
                                                        ,stratify=y
                                                        ,shuffle=True
                                                        , random_state=0)
    df_pred_train=pd.DataFrame()
    df_pred_test=pd.DataFrame()
    
    df_pred_train['id code']=X_train['id code']
    df_pred_train.reset_index(drop=True, inplace=True)
    X_train.drop(['id code'], axis=1, inplace=True)
    
    df_pred_test['id code']=X_test['id code']
    df_pred_test.reset_index(drop=True, inplace=True)
    X_test.drop(['id code'], axis=1, inplace=True)
    
    
    X=df[col1+[tar]]
    
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=see)
    
#    X_test is out of sample
    
#    pred=pd.DataFrame(index=X_test.index)
#    pred_pr=pd.DataFrame(index=X_test.index)
    pred=pd.DataFrame()
    pred_pr=pd.DataFrame()

    pred_raw=pd.DataFrame()
    pred_raw_pr=pd.DataFrame()
    
    
#    pred_v=pd.DataFrame()
#    pred_pr_v=pd.DataFrame()    
    #oversampling
    
    
    
    smo = SMOTE(random_state=0)
    columns = X_train.columns
    os_data_X,os_data_y=smo.fit_sample(X_train, y_train)
    X_train_over = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
    y_train_over=os_data_y['y']
    
    fol=0
    for train_index, test_index in skf.split(X_train_over, y_train_over):
        fol+=1

        clf=logreg.fit(X_train_over.iloc[train_index], np.array(y_train_over.iloc[train_index]))
        # #######---------valid------------------------------------------------------
        X_valid=X_train_over.iloc[test_index]
        y_valid=y_train_over.iloc[test_index]
        
        y_pred_v = clf.predict(X_valid)
        y_pred_proba_v=clf.predict_proba(X_valid)
        
#        pred_v[str(fol)]=y_pred_v
#        pred_pr_v[str(fol)]=y_pred_proba_v[:,1]
#        
        cm=confusion_matrix(y_valid,y_pred_v)
        TN=cm[0,0]
        TP=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        
        '''average in validation'''
        av_auc_v.append(roc_auc_score(y_train_over.iloc[test_index], y_pred_proba_v[:,1]))
        av_gini_v.append(gini_normalized(y_train_over.iloc[test_index], y_pred_proba_v[:,1]))
        av_fn_v.append(FN)
        av_fp_v.append(FP)
        
        
#        print(classification_report(y_test, y_pred))
#        print("1: ", sum(y_pred), "O: ", len(y_pred)-sum(y_pred))
#         #predict that good (0) but he is bad (1)

#        print(c)
#        print("1: ", FP+TP, "O: ", TN+FN)
#        print (cm)    
#        print("!!! Misclassified BAD: {} ({}%)".format(FN, round(FN/cm.sum()*100,3)))
#        print("Misclassified GOOD: {} ({}%)".format(FP, round(FP/cm.sum()*100,3)))
#        print('Accuracy {}%'.format(round(100*(TP+TN)/cm.sum(),2)))
#        print("ROC AUC score: {}".format(  round(roc_auc_score(y_train.iloc[test_index], y_pred_proba[:,1]),4)))
#        print('Gini: {}'.format(  round(gini_normalized(y_train.iloc[test_index], y_pred_proba[:,1]),4)))
#        print('----------------------------------------')
#       
            
#        #######-------pred traw train-----------------------------------------------------
        y_pred_train = clf.predict(X_train)
        y_pred_proba_train=clf.predict_proba(X_train)
        
        pred_raw[str(fol)]=y_pred_train
        pred_raw_pr[str(fol)]=y_pred_proba_train[:,1]
        
        cm=confusion_matrix(y_train,y_pred_train)
        TN=cm[0,0]
        TP=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        
        av_train_auc.append(roc_auc_score(y_train, y_pred_proba_train[:,1]))
        av_train_gini.append(gini_normalized(y_train, y_pred_proba_train[:,1]))
        av_train_fn.append(FN)
        av_train_fp.append(FP)
#        #######-------pred test------------------------------------------------------
        
        if  test_plus_valid==0:
            #        VAlidate on test-------------------------------------------------        
            y_pred_test = clf.predict(X_test)
            y_pred_proba_test=clf.predict_proba(X_test)
            
            pred[str(fol)]=y_pred_test
            pred_pr[str(fol)]=y_pred_proba_test[:,1]
    
#            scor=gini_normalized(y_test, y_pred_proba_test[:,1])
            
            cm=confusion_matrix(y_test,y_pred_test)
            
            TN=cm[0,0]
            TP=cm[1,1]
            FP=cm[0,1]
            FN=cm[1,0]
            
            av_auc.append(roc_auc_score(y_test, y_pred_proba_test[:,1]))
            av_gini.append(gini_normalized(y_test, y_pred_proba_test[:,1]))
            av_fn.append(FN)
            av_fp.append(FP)
        else:   
            #        VAlidate on test+valid-------------------------------------------------
            X_combo=pd.concat([X_test, X_valid], axis=0)
            y_combo=list(y_test)+list(y_valid)
            
            y_pred_combo = clf.predict(X_combo)
            y_pred_proba_combo=clf.predict_proba(X_combo)
            
            pred[str(fol)]=y_pred_combo
            pred_pr[str(fol)]=y_pred_proba_combo[:,1]
    
#            scor=gini_normalized(y_combo, y_pred_proba_combo[:,1])
            
            cm=confusion_matrix(y_combo,y_pred_combo)
            
            TN=cm[0,0]
            TP=cm[1,1]
            FP=cm[0,1]
            FN=cm[1,0]
            
            av_auc.append(roc_auc_score(y_combo, y_pred_proba_combo[:,1]))
            av_gini.append(gini_normalized(y_combo, y_pred_proba_combo[:,1]))
            av_fn.append(FN)
            av_fp.append(FP)
        
#        print("1: ", sum(y_pred), "O: ", len(y_pred)-sum(y_pred))
#        print("1: ", FP+TP, "O: ", TN+FN)
    
#        print(c)
#        print("!!! Misclassified BAD: {} ({}%)".format(FN, round(FN/cm.sum()*100,3)))
#        print("Misclassified GOOD: {} ({}%)".format(FP, round(FP/cm.sum()*100,3)))
#        print('Accuracy {}%'.format(round(100*(TP+TN)/cm.sum(),2)))
#        print("ROC AUC score: {}".format(  round(roc_auc_score(y_test, y_pred_proba_test[:,1]),4)))
#        print('Gini: {}'.format(  round(gini_normalized(y_test, y_pred_proba_test[:,1]),4)))
#             
#        print('----------------------------------------')
#        print('----------------------------------------')
#        
    

    

    
    #        print("1: ", sum(y_pred), "O: ", len(y_pred)-sum(y_pred))
    
#    Averages for metrics
    print('-----valid-----') 
    print('------------min folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.min(av_gini_v),4), round(np.min(av_auc_v),4), round(np.min(av_fn_v),1), round(np.min(av_fp_v),1)))       
    print('------------mean folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.mean(av_gini_v),4), round(np.mean(av_auc_v),4), round(np.mean(av_fn_v),1), round(np.mean(av_fp_v),1)))       
    print('------------std folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.std(av_gini_v),4), round(np.std(av_auc_v),4), round(np.std(av_fn_v),1), round(np.std(av_fp_v),1)) )
    
    print('-----Test-----')
    print('------------min folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.min(av_gini),4), round(np.min(av_auc),4), round(np.min(av_fn),1), round(np.min(av_fp),1)))       
    print('------------mean folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.mean(av_gini),4), round(np.mean(av_auc),4), round(np.mean(av_fn),1), round(np.mean(av_fp),1)))       
    print('------------std folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.std(av_gini),4), round(np.std(av_auc),4), round(np.std(av_fn),1), round(np.std(av_fp),1)) )

    print('-----Train-----')
    print('------------min folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.min(av_train_gini),4), round(np.min(av_train_auc),4), round(np.min(av_train_fn),1), round(np.min(av_train_fp),1)))       
    print('------------mean folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.mean(av_train_gini),4), round(np.mean(av_train_auc),4), round(np.mean(av_train_fn),1), round(np.mean(av_train_fp),1)))       
    print('------------std folds--------------')
    print(" GINI {}  Roc AUC {}  FN {}  FP {}".format(round(np.std(av_train_gini),4), round(np.std(av_train_auc),4), round(np.std(av_train_fn),1), round(np.std(av_train_fp),1)) )

  
    '''------------------------------'''
#    Aggregation of predictions for TEST
    #class

#    y_pred_v=pred_v.min(axis=1)
#    y_pred_v=pred_v.max(axis=1)   
#    median==majority rule
    

    
    y_pred=pred.mean(axis=1)
    y_pred=(y_pred>=0.5)*1

    y_pred_train=pred_raw.mean(axis=1)
    y_pred_train=(y_pred_train>=0.5)*1
    #probability
#    y_pred_pr=pred_pr.min(axis=1)
    
#    y_pred_pr_v=pred_pr_v.mean(axis=1)
    y_pred_pr=pred_pr.mean(axis=1)

    y_pred_train_pr=pred_raw_pr.mean(axis=1)


    '''store'''
    df_pred_train['class']=y_pred_train
    df_pred_train['prob']=y_pred_train_pr
   
    df_pred_test['class']=y_pred
    df_pred_test['prob']=y_pred_pr

    #mean
    #validation score  
    gini_sc_v=np.mean(av_gini_v) 
    roc_sc_v=np.mean(av_auc_v)
    FN_v=np.mean(av_fn_v)
    FP_v=np.mean(av_fp_v)
    
    #test validation score       
    #mean
#    gini_sc=np.mean(av_gini) 
#    roc_sc=np.mean(av_auc)
#    FN=np.mean(av_fn)
#    FP=np.mean(av_fp)

    '''------------------------------'''
    cm=confusion_matrix(y_train,y_pred_train)

    TN=cm[0,0]
    TP=cm[1,1]
    FP=cm[0,1]
    FN=cm[1,0]
    
    print("-----Final on raw Train_set----")
    print (cm)
    print("1: ", FP+TP, "O: ", TN+FN)   
    print("!!! Misclassified BAD: {} ({}%)".format(FN, round(FN/cm.sum()*100,3)))
    print("Misclassified GOOD: {} ({}%)".format(FP, round(FP/cm.sum()*100,3)))
    print('Accuracy {}%'.format(round(100*(TP+TN)/cm.sum(),2)))
    roc_sc_train=roc_auc_score(y_train, y_pred_train_pr)
    print("ROC AUC score: {}".format(  round(roc_sc_train,4)))   
    gini_sc_train=gini_normalized(y_train, y_pred_train_pr)
    print('Gini: {}'.format(round(gini_sc_train,4)))
        
    if  test_plus_valid==0:
        cm=confusion_matrix(y_test,y_pred)
    
        TN=cm[0,0]
        TP=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        
        print("-----Final on Test_set----")
        print (cm)
        print("1: ", FP+TP, "O: ", TN+FN)   
        print("!!! Misclassified BAD: {} ({}%)".format(FN, round(FN/cm.sum()*100,3)))
        print("Misclassified GOOD: {} ({}%)".format(FP, round(FP/cm.sum()*100,3)))
        print('Accuracy {}%'.format(round(100*(TP+TN)/cm.sum(),2)))
        roc_sc=roc_auc_score(y_test, y_pred_pr)
        print("ROC AUC score: {}".format(  round(roc_sc,4)))   
        gini_sc=gini_normalized(y_test, y_pred_pr)
        print('Gini: {}'.format(round(gini_sc,4)))
    else:
        cm=confusion_matrix(y_combo,y_pred)
    
        TN=cm[0,0]
        TP=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        
        print("-----Final on Test_set combo----")
        print (cm)
        print("1: ", FP+TP, "O: ", TN+FN)   
        print("!!! Misclassified BAD: {} ({}%)".format(FN, round(FN/cm.sum()*100,3)))
        print("Misclassified GOOD: {} ({}%)".format(FP, round(FP/cm.sum()*100,3)))
        print('Accuracy {}%'.format(round(100*(TP+TN)/cm.sum(),2)))
        roc_sc=roc_auc_score(y_combo, y_pred_pr)
        print("ROC AUC score: {}".format(  round(roc_sc,4)))   
        gini_sc=gini_normalized(y_combo, y_pred_pr)
        print('Gini: {}'.format(round(gini_sc,4)))

    
    
#    #min
#    #validation score  
#    gini_sc_v=np.min(av_gini_v) 
#    roc_sc_v=np.min(av_auc_v)
#    FN_v=np.min(av_fn_v)
#    FP_v=np.min(av_fp_v)
#    
#    #test validation score       
#    #min
#    gini_sc=np.min(av_gini) 
#    roc_sc=np.min(av_auc)
#    FN=np.min(av_fn)
#    FP=np.min(av_fp)
    return df_pred_train, df_pred_test, y_pred_train, y_pred_train_pr, y_pred, y_pred_pr, gini_sc, roc_sc, FN, FP, gini_sc_v, roc_sc_v, FN_v, FP_v



#for c in df.columns:
#    print("'" +str(c)+ "',")
col1=[
'checking',
'purpose of loan',
#'gross payments',
'savings',
'employed',
'installp',
'marital',
'co-applicant status',
'resident',
'property',
#'age',
'other',
'housing',
'exist credit bureau data',
'job',
'dependents',
'telephon',
'foreign',
'job type_m',

'other_bin',
'job_bin',
'housing_bin',
'job type_mod',
'marital_mod',

'age_l',
'history (number of loans)',
'principal payments_l',
#'gross payments_l',
'loan duration (m)_l',
#'loan duration (m)',
'checking_bin',
'marital_bin',
'history (number of loans)_bin',
'history (number of loans)_mod',
'savings_bin',
'principal payments_bin',
#'age_bin'
'loan duration (m)_bin',
#,
'property_bin'

]




'''------------------------------------------'''

#All in one ohe---------------------------------------------------------

valid=0
split=0.3
folds=6
r=[]
col_nn=[]
col_nn_hot=[]
see=101

gi=1
rez_df_train, res_df_test, rez_train_cl, rez_train_pr, rez_cl, rez_pr, gini_sc, roc_sc, FN, FP, gini_sc_v, roc_sc_v, FN_v, FP_v=logg(df, col1, 1, col_hot, folds, split, see, valid)
print('--------------------------')
print('valid')
print(' GINI: {} Roc AUC {} FN {} FP {}'.format(round(gini_sc_v,4), round(roc_sc_v,4), round(FN_v,0), round(FP_v,0)))
print('test')
print(' GINI: {} Roc AUC {} FN {} FP {}'.format(round(gini_sc,4), round(roc_sc,4), round(FN,0), round(FP,0)))
print('Shape0: {} Shape: {} Folds {} Seed {} Selection: {}'.format(len(col1)+len(col_hot), len(col_nn)+len(col_nn_hot), folds, see, gi))


if gi==1:
    sc=gini_sc
    scor=sc
    sc_v=gini_sc_v
else:
    sc=roc_sc
    scor=sc
    sc_v=roc_sc_v

cont=1
r=[]
#gini_sc
#gini_sc=sc

col_z=col1+col_hot
while  cont==1:  
    rc=''
    cont=0
    ch=[x for x in col_z if x not in r]
#    sc_c=0
    for c in ch:        
        cc=[x for x in ch if x in col1 and x not in r+[c]]
        cc_h=[x for x in ch if x in col_hot and x not in r+[c]]
        
        rez_df_train, res_df_test, rez_train_cl, rez_train_pr, rez_cl, rez_pr, gini_sc, roc_sc, FN, FP, gini_sc_v, roc_sc_v, FN_v, FP_v=logg(df, cc, 1, cc_h, folds, split, see, valid)
        
        #if without c average score is better
        #mod - find 

        if gi==1:
            scor=gini_sc
            scor_v=gini_sc_v
        else:
            scor=roc_sc
            scor_v=roc_sc_v
        
        sc_m=min(sc, sc_v)
        if scor>=sc_m and scor_v>=sc_m :
#        if scor>=sc and scor_v>=sc_v :
            rc=c
            sc=scor
            sc_v=scor_v
#            sc=min(scor, scor_v)
#            sc_v=min(scor, scor_v)
            
    if rc!='':
#        print('\n\n\n\n\n')
#        print (rc)
#        print('\n\n\n\n\n')
        r.append(rc)
        cont=1

print(sc, sc_v, sc_m)


col_nn=[x for x in col1 if x not in r]
col_nn_hot=[x for x in col_hot if x not in r]

'''ээээээээээээээээээээээээээээ'''
rez_df_train, rez_df_test, rez_train_cl, rez_train_pr, rez_cl, rez_pr, gini_sc, roc_sc, FN, FP, gini_sc_v, roc_sc_v, FN_v, FP_v=logg(df, col_nn, 1, col_nn_hot, folds, split, see, valid)
#rez_cl, rez_pr, gini_sc, roc_sc, FN, FP, gini_sc_v, roc_sc_v, FN_v, FP_v=logg(df, cc, 1, cc_h, folds, split, see, valid)
print('--------------------------')
print('valid')
print(' GINI: {} Roc AUC {} FN {} FP {}'.format(round(gini_sc_v,4), round(roc_sc_v,4), round(FN_v,0), round(FP_v,0)))
print('test')
print(' GINI: {} Roc AUC {} FN {} FP {}'.format(round(gini_sc,4), round(roc_sc,4), round(FN,0), round(FP,0)))
print('Shape0: {} Shape: {} Folds {} Seed {} Selection: {}'.format(len(col1)+len(col_hot), len(col_nn)+len(col_nn_hot), folds, see, gi))

p=0
for c in col_nn:
    p+=1
    print('{}. {}'.format(p,c))

p=0
for c in col_nn_hot:
    p+=1
    print('{}. {}'.format(p,c))
    

col_drop=[x for x in col1+col_hot if x not in col_nn+col_nn_hot and x ]
p=0
for c in col_drop:
    p+=1
    print('{}. {}'.format(p,c))
    
    
    
def plot_roc_curve(actual, pred_proba):
    if pred_proba.shape[1]==2:
        pred_proba=pred_proba[:,1]
    fpr, tpr, thresholds = roc_curve(actual, pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba_test[:,1]))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


#sensitivity, recall, hit rate, or true positive rate (TPR)
#True Positive Rate
#TPR =TP/Positive=TP/(TP+FN)=1-FNR
#False Positive Rate
#FPR=FP/Negative=FP/(FP+TN)=1-TNR

def score (pr_d):
    if pr_d==1:
        pr_d=0.999999999999999
    if pr_d==0:
        pr_d=0.000000000000001
    return 20*(np.log2(1/pr_d -1)+10)

def group_score(x):
    if x<0 :
        scg=0
    else:
        scg=int(x/20)
    return scg*20

group_score(score (0.5))


rez_df_test['score']=rez_df_test['prob'].apply(lambda x: group_score(score (x)))
rez_df_train['score']=rez_df_train['prob'].apply(lambda x: group_score(score (x)))

rez_df_test['score']=rez_df_test['prob'].apply(lambda x: score (x))
rez_df_train['score']=rez_df_train['prob'].apply(lambda x: score (x))


rez_df_train['train_test']='train'
rez_df_test['train_test']='test'

rez_df=pd.concat([rez_df_train, rez_df_test], axis=0)
rez_df.to_csv('predict.csv', index=False)


rez_df_train['score'].hist(bins=80, range=[0,400], alpha=0.6, label='train')
rez_df_test['score'].hist(bins=80, range=[0,400], alpha=0.6, label='test')
plt.legend()
plt.savefig('score')
plt.show()


fig, ax1 = plt.subplots()
X=np.arange(0, 400, 1)
def pd_from_score(x):
    return 1/(1+2**(1/20*x-10))
Z=[]
# 1-F(x)- distribution function (1-CDF) Cumulative Distribution Function
a1=rez_df_train['score']
for x in X:
#    Z.append(np.sum(i > x for i in a1)/len(a1))
#    Z.append((a > x).sum()/len(a))
    Z.append(len([1 for i in a1 if i > x])/len(a1))
    
ax1.plot(X,Z,label='Score test')
a=rez_df_test['score']
Z=[]
for x in X:
        Z.append(len([1 for i in a if i > x])/len(a))
        
        
        
ax1.plot(X,Z, label='Score test')   
ax1.plot(X, pd_from_score(X), 'g--', label='Score_pd')
ax1.legend(loc=6)
ax1.grid(b=True, axis='x' , linestyle=':')
'''--'''
ax2 = ax1.twinx()
# probability density function
ax2.hist(a1, histtype='step', range=[0,400], bins=80, alpha=0.8, label='train')
ax2.hist(a, histtype='step', range=[0,400], bins=80, alpha=0.8, label='test')

ax2.legend(loc=7)

plt.xticks(np.arange(0, 400, step=20))
#plt.grid(axis='x',color='r', linestyle='-', linewidth=2)

plt.title("score distribution")
fig.tight_layout() 
plt.savefig('pd_score')
plt.show()

#bad rate
