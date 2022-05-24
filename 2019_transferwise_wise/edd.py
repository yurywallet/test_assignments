# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:25:20 2020

@author: Yury
"""
import os
os.environ['MKL_NUM_THREADS'] = '3' #for core i5 5200
os.environ['OMP_NUM_THREADS'] = '3' #for core i5 5200

import matplotlib.pyplot as plt
import pandas as pd
import gc
import sys
sys.path.append("C://a_machine//TOOLS") #need to be changed
# import encoders as encd
import tools_eda as tool
import tools_graph_nx as grtool

m_fold='C:/a_job/transferwise/EDD homework bundle'



'''====================='''
imp_file='AML homework dataset.csv'
df=pd.read_csv(f'{m_fold}\\{imp_file}')

cols=df.columns
'''====================='''
cur_file='currency.xlsx'
df_cur=pd.read_excel(f'{m_fold}\\{cur_file}', sheet_name='cur_list')

df_ex=pd.read_excel(f'{m_fold}\\{cur_file}', sheet_name='ex_rate')


'''------------------------------------------------'''
tar='flag_transferred'
print(df[tar].value_counts())

a=df['payment_status'].value_counts().reset_index()
a.columns=['payment_status','Number']
a["Share"]=a['Number']/a['Number'].sum()
print(a)

df['rev_flag']=1-df[tar]
# tar_new='flag_new'

# 	recipient_country_code
# !!! 4101	EmailRecipient
# EmptyRecipient
# 22055	BalanceRecipient

# #local???
# 4613	SriLankaLocalRecipient
# 4509	JapaneseLocal
# 4260	BangladeshLocalRecipient
# 7250	VietnamEarthportRecipient
a=df.loc[df['recipient_country_code'].str.contains('Local') ,['recipient_country_code']]

map_cont={'SriLankaLocalRecipient':'LK',
'JapaneseLocal':'JP',
'BangladeshLocalRecipient':'BD',
'VietnamEarthportRecipient':'VN'}

df['recipient_country_code']=df['recipient_country_code'].apply(lambda x: map_cont[x] if x in map_cont.keys() else x )




# 2 letters to 3 letters
df=pd.merge(df,df_cur[['CountryCode','CountryCode3']],left_on='recipient_country_code', right_on='CountryCode', how='left')
df['recipient_country_code']=df['CountryCode3']
df.drop(['CountryCode3','CountryCode'], axis=1, inplace=True)


# 'sending_bank_country'
map_2_3=df_cur[['CountryCode','CountryCode3']].set_index('CountryCode').to_dict()['CountryCode3']
df['sending_bank_country']=df['sending_bank_country'].apply(lambda x: map_2_3[str(x)] if str(x) in map_2_3.keys() else x )



# 'addr_city'
df['addr_city']=df['addr_city'].str.lower()


#nan to -1
for c in ['transfer_sequence', 'days_since_previous_req']:
    df[c].fillna(-1, inplace=True)


#to date
date_cols=['date_user_created','date_request_submitted', 'date_request_received', 'date_request_transferred'
          ,'date_request_cancelled', 'first_attempt_date', 'first_success_date']
for c in date_cols:
    df[c]=pd.to_datetime(df[c], format='%d/%m/%Y %H:%M')


for c in ['invoice_value', 'invoice_value_cancel']:
    df[c].fillna(0, inplace=True)

'''========================================================='''
df_cancelled=df.loc[df[tar]==0]

user_cancelled=df_cancelled.groupby(['user_id'])['target_recipient_id'].size().reset_index()

'''========================================================='''


''' Clean data '''

#remove transactions with no money received
# date_request_received​  - Date at which we received the customer’s money  
#1-----NoMoney
c='F_no_money'
df=df.loc[pd.notnull(df['date_request_received'])]
'''not defined source'''
#2------ payment_type​  - Payment method used to upload money
df=df.loc[pd.notnull(df['payment_type'])]

#3------ zero sum invoice
df['F_value']=df[['invoice_value', 'invoice_value_cancel']].sum(axis=1)
df=df.loc[df['F_value']>0]

#4------ must know what country to send
df=df.loc[pd.notnull(df['recipient_country_code'])]

df.loc[df[tar]==0].shape[0]/100000


print(f"Cancelled in data: {round(100*df['rev_flag'].sum()/df.shape[0],2)} %")
a=df['payment_status'].value_counts().reset_index()
a.columns=['payment_status','Number']
a["Share"]=a['Number']/a['Number'].sum()
print(a)

df_nm=df.loc[df['sending_bank_country']=='Other/unknown']
df_nm=df.loc[pd.isnull(df['recipient_country_code'])]
df_nm=df.loc[df['transfer_to_self']=='N.A. Recipient Email Unknown']

# 'N.A. Recipient Email Unknown'

'''========================================================='''

c='payment_reference_classification'
a=tool.uni(df,c)





''' ------------------ new features -----------'''


df['F_isBiz']=(df['flag_personal_business']=='Business')*1
df['F_isBiz'].sum()

# USD EUR oposite
c='ccy_send'
df=pd.merge(df, df_ex[['Currency code','Units per EUR']], left_on=c, right_on='Currency code', how='left')
df.drop('Currency code', axis=1, inplace=True)
df['F_value_EUR']=df['F_value']/df['Units per EUR']

a=df[['F_value','ccy_send','ccy_target','Units per EUR', 'F_value_EUR']]


# inside country
c1='recipient_country_code'
c2='addr_country_code'
df['F_local_trans']=(df[c2]==df[c1])*1
df['F_local_trans'].sum()

c1='recipient_country_code'
c2='addr_country_code'
df['F_local_trans']=(df[c2]==df[c1])*1


#how many transfers done by user
df['F_user_count']=df.groupby('user_id')['user_id'].transform('count')
a=tool.uni(df,'F_user_count')

a=df.loc[df['F_user_count']>1]
user_2plus=a['user_id'].unique()

#to how many countries
c='F_user_num_countries'
df[c]=df.groupby('user_id')['recipient_country_code'].transform('nunique')


# transfer to new location
c='F_user_new_loc'
df[c]=1 #first is always new
df.loc[df['F_user_count']>1,c]=df.loc[df['F_user_count']>1,['user_id','date_request_submitted','recipient_country_code']].apply(lambda x: 0 if x[2] in df.loc[(df['user_id']==x[0]) & (df['date_request_submitted']<pd.to_datetime(x[1], format='%d/%m/%Y %H:%M')), 'recipient_country_code'].unique() else 1, axis=1)

# a=df.loc[df['F_user_count']>1,['user_id','date_request_submitted','recipient_country_code',c]]
    
# Number of different countries involved 
#Example: SHE to GEO via CYP
df['F_num_country']=df[['addr_country_code','recipient_country_code','sending_bank_country']].nunique(1)
tool.uni(df,'F_num_country')

#region--------------------------------------------------
#sender
col_geo=['Region', 'continent', 'high_risk', 'high_risk_tax', 'g10', 'g24plus']
c='addr_country_code'
df=pd.merge(df, df_cur[['CountryCode3']+col_geo], left_on=c, right_on='CountryCode3', how='left')
for cc in col_geo:
    df.rename(columns={cc: f'F_send_{cc}'}, inplace=True)
df.drop(['CountryCode3'], axis=1, inplace=True)


# receiver
c='recipient_country_code'
df=pd.merge(df, df_cur[['CountryCode3']+col_geo], left_on=c, right_on='CountryCode3', how='left')
for cc in col_geo:
    df.rename(columns={cc: f'F_receiv_{cc}'}, inplace=True)
df.drop(['CountryCode3'], axis=1, inplace=True)

# bank
c='sending_bank_country'
df=pd.merge(df, df_cur[['CountryCode3']+col_geo], left_on=c, right_on='CountryCode3', how='left')
for cc in col_geo:
    df.rename(columns={cc: f'F_bank_{cc}'}, inplace=True)
df.drop(['CountryCode3'], axis=1, inplace=True)

df['F_bank_continent'].fillna('Not_defined', inplace=True)
df['F_bank_Region'].fillna('Not_defined', inplace=True)

df[[c, 'F_bank_high_risk']]
#Fillna




#G10 currencies
g10_cur=['USD','EUR','GBP','JPY','AUD','NZD','CAD','CHF','NOK','SEK']
#indicator
#sender
c='F_g10_send_cur'
df[c]=0
df.loc[df['ccy_send'].isin(g10_cur),c]=1
df[c].sum()
# receiver
c='F_g10_receiv_cur'
df[c]=0
df.loc[df['ccy_target'].isin(g10_cur),c]=1
df[c].sum()

#groupping: g10 or other
df['F_ccy_target_gr10']=df[['ccy_target','F_g10_receiv_cur']].apply(lambda x: x[0] if x[1]==1 else 'OTHER',axis=1)
df['F_ccy_send_gr10']=df[['ccy_send','F_g10_send_cur']].apply(lambda x: x[0] if x[1]==1 else 'OTHER' , axis=1)


# Convertion
df['F_isConvertion']=(df['ccy_target']!=df['ccy_send'])*1 #99% with currenct change
df['F_ccy_pair']=df['ccy_send']+'_'+df['ccy_target']
df['F_ccy_pair_gr10']=df['ccy_send'] +'_'+df['F_ccy_target_gr10']

a=tool.uni(df,'F_ccy_pair_gr10')

#least developed countries


'''=================='''

# official currency of the country



#target
c2='F_ccy_target_off'
c1='recipient_country_code'
df=pd.merge(df, df_cur[['CountryCode3','CurrencyCode']], left_on=c1, right_on='CountryCode3', how='left')
df.rename(columns={'CurrencyCode': c2}, inplace=True)
df.drop(['CountryCode3'], axis=1, inplace=True)

a=tool.uni(df,c1)


'''=================='''
#dates

df['F_date_subm2user_created']=(df['date_request_submitted']-df['date_user_created']).dt.days
a=df.loc[df['F_date_subm2user_created']<0]

df['F_date_subm2received']=(df['date_request_submitted']-df['date_request_received']).dt.days
a=df.loc[df['F_date_subm2received']<0]

df['F_date_created2first_attemp']=(df['first_attempt_date']-df['date_user_created']).dt.days
a=df.loc[df['F_date_subm2received']<0]


df['F_date_received2transfered']=(df['date_request_transferred']-df['date_request_received']).dt.days
a=df.loc[df['F_date_received2transfered']<0]


#sequence
c='transfer_sequence'
a=df.loc[df[c]>2]
n=1
df[f'F_after_{n}_trans']=0
df.loc[df[c]>n,f'F_after_{n}_trans']=1
a=tool.uni(df,f'F_after_{n}_trans')

for n in range(1,4):
    df[f'F_after_{n}_trans']=0
    df.loc[df[c]>n,f'F_after_{n}_trans']=1
    a=tool.uni(df,f'F_after_{n}_trans')


#users that are in sender and recipient
c='F_user_2parties'
df[c]=0
df.loc[df['user_id'].isin(df['target_recipient_id'].unique()), c]=1
a=df.loc[df[c]>1]
a=tool.uni(df,c)



''' ------------------ ----- -----------'''

print(f'''
Countries
{df['addr_country_code'].nunique()} address countries 
{df['recipient_country_code'].nunique()} destination countries
Currencies
{df['ccy_send'].nunique()} sender currency
{df['ccy_target'].nunique()} target currency
{df['F_ccy_pair'].nunique()} currency pairs
Users
{len(set(list(df['user_id'].unique())+list(df['target_recipient_id'].unique())))} total
{df['user_id'].nunique()} senders
{df['target_recipient_id'].nunique()} recipients
Accounts
{df.loc[df['flag_personal_business']=='Business','flag_personal_business'].shape[0]}  business
{df.loc[df['flag_personal_business']=='Personal','flag_personal_business'].shape[0]}  personal
Payment types
{df['payment_type'].nunique()} different payment types
Banks
{df['sending_bank_name'].nunique()} unique counterparty banks from
{df['sending_bank_country'].nunique()} countries
Devices
{df['device'].nunique()} different device types
Time
Users were created from {df['date_user_created'].min()} till {df['date_user_created'].max()} 
Requests submitted from {df['date_request_submitted'].min()} till {df['date_request_submitted'].max()} 
''')




'''-------------------------Outliers --------------------'''
# invoice value


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
    
cl=['user_id',	'target_recipient_id',	'request_id','ccy_send','ccy_target','transfer_sequence', 'F_isBiz', 'F_value']
c='F_value'
a, a_extreme, std3, q99 =stat_draw(df.loc[df['flag_transferred']==1],c,cl, he=10)

a_invValue_3std =df.loc[df[c]>std3, cols]


#---------
c='transfer_sequence'
a, a_extreme, std3, q99 =stat_draw(df.loc[df['flag_transferred']==1],c,cl, he=10)
a_transSeq_3std =df.loc[df[c]>std3, cols]

#---------
cl=['user_id',	'target_recipient_id',	'request_id','ccy_send','ccy_target','transfer_sequence', 'F_isBiz', 'F_value_EUR']
c='F_value_EUR'
a, a_extreme, std3, q99 =stat_draw(df.loc[df['flag_transferred']==1],c,cl, he=10)
# a_invValue_3std =df.loc[df[c]>std3, cols]


# extreme number of transfers
df_2plus_agg=df.groupby('user_id')['target_recipient_id'].agg(['size','nunique']).reset_index()
df_2plus_agg.columns=['user_id', '#transfers', '#unique_targets']

a, a_extreme, std3, q99=stat_draw(df_2plus_agg,'#transfers',['user_id', '#transfers', '#unique_targets'], 
                       he=10)


'''------------------------- LINKs----------------- '''

#from users with more than 1 transfer
# df_2plus_agg=df.loc[df['user_id'].isin(user_2plus)].groupby('user_id')['target_recipient_id'].agg(['size','nunique']).reset_index()

# df_2plus=df.loc[df['user_id'].isin(user_2plus)]

# How many transfers is made by user and to how many users


#number of inflows before this


#if cancelled not removed will show the possible connection
#if removed cancelled - only actual transfers accepted
df_links= grtool.link_graph(df[['user_id', 'target_recipient_id', 'F_value_EUR']])





'''-------------------------MAX PATH LENGTH '''



df_groups=grtool.connected_nodes(df[['user_id', 'target_recipient_id', 'F_value_EUR']])

print("Group size Medians: ", df_groups['F_groups_size'].median(), " | 99 quantile: ", df_groups['F_groups_size'].quantile(0.99))

print("Group turnover Medians: ", round(df_groups['F_groups_turnover'].median(),2), " | 99 quantile: ", round(df_groups['F_groups_turnover'].quantile(0.99),2))

aa=round(df_groups['F_groups_turnover'].quantile(0.001),2)

a=df.loc[df['F_value_EUR']<1]
a.shape[0]

a=df_groups['F_groups_size'].value_counts().reset_index()
a.columns=['F_groups_size',	'Number_of_groups']
a.sort_values([	'Number_of_groups', 'F_groups_size'], ascending=[False,True], inplace=True)

a.tail(10)
print ("Groups that have more than 6 members: ", a.loc[a['F_groups_size']>6, 'Number_of_groups'].sum())



aa=['69cf237499d8ccac9211602c37807d92',
 'cd654f1f3e98b806281200c0c6727323']
a=df.loc[(df['user_id'].isin(aa)) | (df['target_recipient_id'].isin(aa)),
         ['user_id','target_recipient_id', 'request_id', 'ccy_send', 'F_value']
         ]

''' ---- #plot social graph!!! --------------------------------------------'''


#!!! takes a lot of time to plot !!!
# mapping_w=grtool.plot_digraph(df[['user_id', 'target_recipient_id', 'F_value_EUR']],
#                       weight=True, pixels=9000, dpi=100, remap=True)


gc.collect()

# filter out users with more than n links
n=2
df_links_filtr=df_links.loc[df_links['links_num']>n]

# just connections
# mapping=grtool.plot_digraph(df.loc[df['user_id'].isin(df_links_filtr['user_id'].unique()),['user_id', 'target_recipient_id', 'F_value_EUR']],
#                      weight=False, pixels=9000, dpi=100, remap=True)

# with weight - thickness is defined by transfer value
mapping_w=grtool.plot_digraph(df.loc[df['user_id'].isin(df_links_filtr['user_id'].unique()),['user_id', 'target_recipient_id', 'F_value_EUR']],
                     weight=True, pixels=9000, dpi=100, remap=True)


'''----------------------Rules-----------------------'''
# EmptyRecipient
# 22055	BalanceRecipient
# c='rule_1_email'
# df[c]=df[['days_since_previous_req','recipient_country_code']].apply(lambda x: 1 if x[0]<=0 and x[1]=='EmailRecipient' else 0, axis=1)



'''--------EDA-----------------'''
df_card=df.loc[df['payment_type']=='Cards']
df_bank=df.loc[df['payment_type']=='Bank Transfer']
df_other=df.loc[(df['payment_type']!='Bank Transfer') & (df['payment_type']!='Cards')]

X_card=tool.summary_df(df_bank)
X_bank=tool.summary_df(df_card)
X=tool.summary_df(df)

#unique values
df['user_id'].value_counts()
print(df[c].value_counts())

c='payment_status'
print(df[c].value_counts())


c='transfer_sequence'
print(df[c].value_counts())

# currency
c='ccy_send'
print(df[c].value_counts())
c='ccy_target'
print(df[c].value_counts())




# 'transfer_sequence' | 'days_since_previous_req'

a=df[df['user_id'].isin(df['target_recipient_id'].unique())]


for c in cols[3:]:
    print(c)
    a=tool.uni(df,c,labels=['ALL', 'Cancelled'], limit=25)

#----------------------------------------
colz=['F_date_subm2user_created'
,'F_date_subm2received'
,'F_date_created2first_attemp'
,'F_date_received2transfered' #-1 19 cases
,'F_isConvertion'
,'F_ccy_pair'
,'F_local_trans'
# ,'F_g10_send_cur'
# ,'F_g10_reciev_cur'
,'F_receiv_high_risk' #!!
,'F_receiv_high_risk_tax' 
,'F_receiv_g10'
,'F_receiv_g24plus'
,'F_receiv_Region'
,'F_receiv_continent' #Africa
,'F_send_high_risk' #!!
,'F_send_high_risk_tax' 
,'F_send_g10' #!
,'F_send_g24plus'
,'F_send_Region' #!!
,'F_send_continent' #Africa, Asia
,'F_after_2_trans'
,'F_after_3_trans'
,'F_user_count'
,'F_user_num_countries'
,'F_date_subm2user_created'
,'F_date_subm2received'
,'F_date_created2first_attemp'
,'F_date_received2transfered' #-1 19 cases
,'F_isConvertion'
,'F_ccy_pair_gr10'
,'F_local_trans'
# ,'F_g10_send_cur'
# ,'F_g10_reciev_cur'
,'F_receiv_high_risk' #!!
,'F_receiv_high_risk_tax' 
,'F_receiv_g10'
,'F_receiv_g24plus'
,'F_receiv_Region'
,'F_receiv_continent' #Africa
,'F_send_high_risk' #!!
,'F_send_high_risk_tax' 
,'F_send_g10' #!
,'F_send_g24plus'
,'F_send_Region' #!!
,'F_send_continent' #Africa, Asia
,'F_user_count'
,'F_user_num_countries']


for c in colz:
    print(c)
    a=tool.woe_distr(df,c, 'rev_flag', alf=0.5, prnt=False)
    tool.draw_distr(a,c,labels=['ALL', 'Cancelled'])
    




# a=df.loc[df[c]<0]

# request_id=['e9b530f7bb7bd42316af03b66a17a588', '7b5b4dba87ddf70634c1564bdf331bc5', 'edd318ffc1845473be067ee8359a45c3',
#             '435d6ab1ba16ba7e05e09d9728bc36ca','fbbe5e8023cc9e9eb575aa1b930d949e',
#             '2548a4ac7ad6eddd035bced24ec6d964']
# a=df.loc[df['request_id'].isin(request_id),['request_id']+date_cols]


# a=df.loc[(df['user_id'].isin(['86b73eb2029f427076f9fc3449acead1'])) | ( df['target_recipient_id'].isin(['86b73eb2029f427076f9fc3449acead1'])) ]
# different rules by country

# chains of payments 

#last transfer or user
    
#----------- High risk countries --------------
cc=['user_id', 
    'request_id',
'addr_country_code',
'sending_bank_country',
'recipient_country_code',
'payment_type',
'ccy_send',
'ccy_target',
 'F_value',	
 'F_isBiz',
'transfer_sequence',
'days_since_previous_req',
 'payment_status'
]



#----------- High risk countries --------------
# SENDer & REceiver
a_b=df.loc[(df['F_send_high_risk']==1) & (df['F_receiv_high_risk']==1)  ,cc]

a_HRL=df.loc[(df['F_send_high_risk']==1) | (df['F_receiv_high_risk']==1),cols]
a_HRL.to_excel(f"{m_fold}/res_HighRiskLocation.xlsx", index=False)


print(f'''Out of {a_HRL.shape[0]} transfers that have high-risk location at least as one side 
- {a_b.shape[0]} have high-risk locations as both sides,
- {df.loc[(df['F_send_high_risk']==1)].shape[0]} were sent and 
- {df.loc[(df['F_receiv_high_risk']==1)].shape[0]} were recieved by user from such country
''')

print(f'''Percent of data to monitor:  {round(100*a_HRL.shape[0]/df.shape[0],2)} % ({a_HRL.shape[0]} transfers)''')

#----------------------------------------------------------------

'''---------------- K-MEANS clustering ---------------'''
from sklearn.cluster import KMeans
# from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import tools_OHE as OHE
#---------------------
#--------------------- Test ideas ----------------
  
gc.collect()

df_clust=df[[
    # 'F_isConvertion', 
'F_local_trans',
'F_isBiz',
'transfer_sequence',
# 'F_ccy_pair_gr10',
'F_user_2parties',
'device',
'F_send_Region', #!!
'F_send_continent',
'F_send_high_risk',
'F_value_EUR']].set_index(df['user_id'])


df_ohe, cols_ohe=OHE.one_hot_encoder_orig(df_clust)

# ,
# 'F_g10_reciev_cur',
# 'F_g10_send_cur'

df_ohe.fillna(0, inplace=True)
cols_ohe=df_ohe.columns



stds=StandardScaler().fit(df_ohe)
df_ohe = stds.transform(df_ohe)


#Elbow-plot
sum_of_squared_distances = []
# a=pd.DataFrame(columns=['#clusters', 'S_score', 'CH_score'])
K=range(4,15)
for k in K:
    kmeans = KMeans(n_clusters=k)
    model = kmeans.fit(df_ohe)
    sum_of_squared_distances.append(kmeans.inertia_)
    labels = kmeans.labels_
    # S_s=metrics.silhouette_score(df_ohe, labels, metric = 'euclidean')
    # CH_s=metrics.calinski_harabasz_score(df_ohe, labels)
    # a.loc[a.shape[0]]=[k, S_s, CH_s]


plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('sum_of_squared_distances')
plt.title('elbow method for optimal k')
plt.show()

# ----------------

# tool.uni(df,'transfer_to_self')
#--------------------------------------------------------

set0=[
'payment_type',
# 'ccy_send',
# 'ccy_target',
'transfer_to_self',
# 'sending_bank_country',
# 'payment_reference_classification',
# 'device',
'transfer_sequence',
'days_since_previous_req',

'F_isBiz',
# 'F_value',
'F_value_EUR',

# 'F_local_trans',

'F_send_Region',
'F_send_continent',
'F_send_high_risk',
# 'F_send_high_risk_tax',
# 'F_send_g10',
# 'F_send_g24plus',
'F_receiv_Region',
'F_receiv_continent',
'F_receiv_high_risk',

'F_bank_Region',
'F_bank_continent',
# 'F_bank_high_risk'
# 'F_bank_high_risk_tax',
# 'F_bank_g10',
# 'F_bank_g24plus',

# 'F_receiv_high_risk_tax',
# 'F_receiv_g10',
# 'F_receiv_g24plus',
# 'F_g10_send_cur', #G10 currencies 0 or 1
# 'F_g10_receiv_cur',
'F_ccy_target_gr10', #G10 currency or OTHER
'F_ccy_send_gr10',
# 'F_isConvertion',
# 'F_ccy_pair',
# 'F_ccy_pair_gr10',
# 'F_ccy_target_off',
# 'F_after_1_trans',
# 'F_after_2_trans',
# 'F_after_3_trans',
# 'F_user_count', #number of transfers by user in data
# 'F_user_num_countries', #to how many countries
'F_user_2parties',
'F_num_country' # how many countries involved in transaction
]

# df['F_bank_continent'].describe()
# df['F_bank_continent'].unique()

# set1=[
#     # 'F_isConvertion', 
# 'F_local_trans',
# 'F_isBiz',
# 'transfer_sequence',
# 'days_since_previous_req',
# # 'F_ccy_pair_gr10',
# 'F_user_2parties',
# 'device',
# 'F_send_Region', #!!
# 'F_send_continent',
# 'F_send_high_risk',

# 'F_receiv_high_risk' #!!
# ,'F_receiv_Region'
# ,'F_receiv_continent'
# , 'F_value_EUR'
# ]

# set2=[
 
# 'F_local_trans',
# 'F_isBiz',
# 'transfer_sequence',
# 'days_since_previous_req',

# 'F_user_2parties',
# 'device',
# 'F_bank_Region',
# 'F_bank_high_risk', 

# 'F_send_Region', 
# 'F_send_continent',
# 'F_send_high_risk',

# 'F_receiv_high_risk', #!!
# 'F_receiv_Region',
# 'F_receiv_continent',

# 'F_value_EUR'

# ]

#--------------------------------------------------------

df_clust=df[set0].set_index(df['user_id']).copy()

#--- OHE -------------

df_ohe, cols_ohe=OHE.one_hot_encoder_orig(df_clust)
df_ohe.fillna(0, inplace=True)
cols_ohe=df_ohe.columns
stds=StandardScaler().fit(df_ohe)
df_ohe = stds.transform(df_ohe)

#--------------------------------------------------------

#density based clustering
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

bw=0.8
bandwidth = estimate_bandwidth(df_ohe, quantile=bw, n_samples=500, random_state=1)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(df_ohe)
labels = ms.labels_

#centers
cluster_centers = ms.cluster_centers_
df_clust_centr_MS=pd.DataFrame(columns=['Cluster'] +cols_ohe.to_list())
for  c in range(len(cluster_centers)):
    df_clust_centr_MS.loc[df_clust_centr_MS.shape[0]]=[c]+list(stds.inverse_transform(cluster_centers[c]))
df_clust_centr_MST=df_clust_centr_MS.T

# unique columns values as center coordinates
df_clust_centr_MST['uniq']=df_clust_centr_MST.nunique(axis=1)
a=df_clust_centr_MST['uniq']

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Number of estimated clusters : %d" % n_clusters_)

aa=pd.Series(labels).value_counts().reset_index()
aa.columns=['#cluster','#users']
print(aa)


l1=aa.loc[aa['#users']==1, '#cluster']
l2=aa.loc[aa['#users']==2, '#cluster']
l3=aa.loc[aa['#users']==3, '#cluster']



df[f'label_MShift_{bw}']=labels


a=df.loc[df[f'label_MShift_{bw}'].isin(l1), cc+[f'label_MShift_{bw}']].sort_values([f'label_MShift_{bw}'])
a2=df.loc[df[f'label_MShift_{bw}'].isin(l2), cc+[f'label_MShift_{bw}']].sort_values([f'label_MShift_{bw}'])
a3=df.loc[df[f'label_MShift_{bw}'].isin(l3), cc+[f'label_MShift_{bw}']].sort_values([f'label_MShift_{bw}'])

a_3=df.loc[df[f'label_MShift_{bw}'].isin(aa.loc[aa['#users']<=3, '#cluster']), cc+[f'label_MShift_{bw}']].sort_values([f'label_MShift_{bw}'])

print("To Review :", round(a_3.shape[0]/df.shape[0],4))

#---------------------------------------------
#---------------------------------------------
#---------------------------------------------
n=int(n_clusters_)
# n=125
kmeans = KMeans(n_clusters=n, max_iter=300, n_init=30, random_state=1)
kmeans.fit(df_ohe)


labels = kmeans.predict(df_ohe)
# sum([1 for x in labels if x==18])

centroids= kmeans.cluster_centers_

df_clust_centr=pd.DataFrame(columns=['Cluster'] +cols_ohe.to_list())
for  c in range(len(centroids)):
    df_clust_centr.loc[df_clust_centr.shape[0]]=[c]+list(stds.inverse_transform(centroids[c]))
df_clust_centrT=df_clust_centr.T


df[f'label_Kmean_{n}']=0
df[f'label_Kmean_{n}']=labels

aa=pd.Series(labels).value_counts().reset_index()
aa.columns=['#cluster','#users']
print(aa)

# a=df.loc[df[f'label_Kmean_{n}']==30]
akm_3=df.loc[df[f'label_Kmean_{n}'].isin(aa.loc[aa['#users']<=16, '#cluster']), cc+[f'label_Kmean_{n}']].sort_values([f'label_Kmean_{n}'])

print("To Review :", round(akm_3.shape[0]/df.shape[0],4))



#-------------------------------------------------------------
# Common by 2 methods
akm_u=akm_3['user_id'].unique()
a_u=a_3['user_id'].unique()

common_users=set(a_u).intersection(akm_u)
print('Common users by two clustering:', len(common_users))
a_common=df.loc[df['user_id'].isin(common_users), cc]



#---------------------------------------------
# SAVE RESULTS



#save to excel
writer = pd.ExcelWriter(f"{m_fold}/res_Risky2Monitor.xlsx", engine='xlsxwriter')
workbook=writer.book

#summary
X.to_excel(writer, sheet_name='Fields', index=False)
worksheet = writer.sheets['Fields']
worksheet.set_column(0, 0, 30)
worksheet.set_column(1, 7, 15)
#describe
# l=0
# # worksheet = workbook.add_worksheet('Describe')
# for c in cols:
#    # worksheet.write(l, 0, c)
#    df[c].describe().reset_index().to_excel(writer,sheet_name='Describe',startrow=1+l*9 , startcol=0, index=False) 
#    l+=1
# worksheet = writer.sheets['Describe']
# worksheet.set_column(0, 3, 30)

#invoice value
a_invValue_3std.to_excel(writer, sheet_name='S3_InvValue', index=False)

#trans sequence
a_transSeq_3std.to_excel(writer, sheet_name='S3_TransSeq', index=False)


#Clusters
#kmeans
akm_3.to_excel(writer, sheet_name='S5_KMeans_clust', index=False)
df_clust_centrT.to_excel(writer, sheet_name='S5_KMeans_clust_centers', index=True)

#MeanShift
a_3.to_excel(writer, sheet_name='S5_MeanShift_clust', index=False)
df_clust_centr_MST.to_excel(writer, sheet_name='S5_MeanShift_clust_centers', index=True)

a_common.to_excel(writer, sheet_name='S5_Common', index=False)

#High-risk location
a_HRL.to_excel(writer, sheet_name='S6_HighRiskLoc', index=False)
# Close the Pandas Excel writer and output the Excel file.
writer.save()

# df['user_id'].describe().reset_index()
# df.loc[df['addr_country_code']=='ESP','F_value_EUR'].describe()


