# -*- coding: utf-8 -*-
"""

@author: Yury Koshelyuk (https://www.linkedin.com/in/yurywallet/)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
import seaborn as sns

plt.figure(figsize=(20, 3))
plt.rc_context(
    {'text.color':'darkslategrey',
     'axes.labelcolor':'steelblue',
     'axes.edgecolor':'darkslategrey', 
     'xtick.color':'darkslategrey', 
     'ytick.color':'darkslategrey',
     'figure.facecolor':'white'})
# plt.set_cmap("YlGnBu")

fpath="C:\\a_job\\bolt\\BusinessAnalyst"
df_ord=pd.read_csv(f'{fpath}\\orders.csv')
df_ord.info()
df_ord.columns.to_list()
tmp=df_ord.head(1000)


df_tic=pd.read_csv(f'{fpath}\\tickets.csv')
df_tic.sort_values(	'ticket_created_at', inplace=True)
df_tic.columns.to_list()

tmp=df_tic.head(1000).copy()


'''----------------------------------------'''
rid='rider_id'
did='driver_id'


#checks in the data
print('Missing (null) values\n',df_ord.isnull().sum())
df_ord[did].isnull().sum()/df_ord.shape[0]

print(f'''Tickets created from {df_tic['ticket_created_at'].min()} to {df_tic['ticket_created_at'].max()}''')
print(f'''Orders created from {df_ord['created'].min()} to {df_ord['created'].max()}''')

df_ord.loc[df_ord['created']==df_ord['created'].max()]
df_ord.loc[df_ord['created']>="2020-01-31"]
df_ord=df_ord.loc[df_ord['created']<"2020-01-31"].copy()


print(f"Tickets were created by {list(df_tic['user_type'].unique())}")



# df_tic.loc[df_tic['ticket_created_at']==df_tic['ticket_created_at'].max()]
# df_ord=df_ord.loc[df_ord['created']!=df_ord['created'].max()].copy()


'''----------------------------------------'''

# tmp['tmp']=(pd.to_datetime(tmp['ticket_first_responded_at'])-pd.to_datetime(tmp['ticket_created_at'])).dt.total_seconds()/3600
# tmp['tmp']=(pd.to_datetime(tmp['ticket_last_solved_at'])-pd.to_datetime(tmp['ticket_created_at'])).dt.total_seconds()/3600
# tmp['tmp']=(pd.to_datetime(tmp['ticket_last_solved_at'])-pd.to_datetime(tmp['ticket_first_responded_at'])).dt.total_seconds()/3600
  
 # 'first_response_time'='ticket_first_responded_at'-'ticket_created_at'
'''----------------------------------------'''


a=df_tic.groupby(['user_type'])[['first_response_time',
 'total_resolution_time']].mean()
a=df_tic.groupby(['user_type'])[['first_response_time',
 'total_resolution_time']].std()


a=df_tic.groupby(['is_incoming'])[['first_response_time',
 'total_resolution_time']].mean()

a=df_tic[['user_type','is_incoming']].drop_duplicates()



df_ord['completed'].value_counts()/df_ord.shape[0]

df_ord['rating'].value_counts()/df_ord.shape[0]

c1='solve_respons_time'
df_tic[c1]=(pd.to_datetime(df_tic['ticket_last_solved_at'])-pd.to_datetime(df_tic['ticket_first_responded_at'])).dt.total_seconds()/3600
df_tic[c1].describe()

#unique drivers and riders
print(f"There are {len(df_ord[did].unique())} unique drivers and {len(df_ord[rid].unique())} unique riders in orders table")
a1=df_ord[rid].value_counts().reset_index().head(10)
# a1['index']=a1['index'].astype('str')
print("--------------\n",a1)
a2=df_ord[did].value_counts().reset_index().head(10)
print("--------------\n",a2)

# df_ord.loc[df_ord['created']==df_ord['created'].max()]

# df_ord=df_ord.loc[df_ord['created']!=df_ord['created'].max()].copy()

def date_parsing(df, col='Date'):
    df[col] = pd.to_datetime(df[col])
    
    df['ymd'] = df[col].dt.strftime('%Y%m%d')
    #day of the week
    df['WeekDay']=df[col].dt.weekday #0=Monday
    # df['WeekDay']=df['Date'].dt.strftime('%w') #0=Sunday
    df['WeekDayName'] = df[col].dt.strftime('%a')
    # day_names=['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']
    # day_names=day_names[::-1]
    day_names=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['WeekNum']=df[col].dt.week #starts from Monday
    '''
    %U week number of year, with Sunday as first day of week (00..53).
    %V ISO week number, with Monday as first day of week (01..53).
    %W week number of year, with Monday as first day of week (00..53).
    '''
    #df['Year_Week'] = df[col].dt.strftime('%Y_%V') #starts from Monday
    df['Year_Week'] = df[col].dt.strftime('%G_%V') #starts from Monday

    return df
 
    

    
day_names=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def datetime_parsing(df, col='Date'):
 
    df[col] = pd.to_datetime(df[col])
    #hour
    df['Hour']=df[col].dt.hour
    df['WeekDay_Hour'] = df[['WeekDay','Hour']].apply(lambda x : f'{x[0]}_{x[1]}', axis=1)
    df['WeekDayName_Hour'] = df[['WeekDayName','Hour']].apply(lambda x : f'{x[0]}_{x[1]}', axis=1)

    return df 

df_ord=date_parsing(df_ord,col='created')
df_ord=datetime_parsing(df_ord,col='created')

df_tic=date_parsing(df_tic,col='ticket_created_at')
df_tic=datetime_parsing(df_tic,col='ticket_created_at')


c='Year_Week'
a=df_tic.groupby(c).size().reset_index()
b=df_ord.groupby(c).size().reset_index()
tmp=pd.merge(a,b, how="left", on=c)
tmp.columns=[c,'tickets', 'rides']
tmp['tpr']=tmp['tickets']/tmp['rides']


tmp=df_tic.groupby('WeekDayName').size()
#a=df_ord.loc[df_ord['id'].isin([184515980773287911])]
#[x for x in df_tic['id'] if x in df_ord['id']]



''' ----------------------------------------- '''
#-- FIND FULL WEEKS
a=df_ord.groupby(['Year_Week', 'WeekDay'])['id'].size().reset_index()
a.columns=['Year_Week', 'WeekDay', 'Num_orders']

a1=a.groupby(['Year_Week'])['WeekDay'].size().reset_index()
a2=a.groupby(['Year_Week'])['Num_orders'].mean().reset_index()

# tmp=df_ord.loc[df_ord[	'Year_Week']=='2020_01', 'ymd'].unique()
tmp=pd.merge(a1, a2, how='left', on='Year_Week' )
# fullweek
tmp=tmp.loc[tmp['WeekDay']==7]


a=df_tic.groupby(['Year_Week', 'WeekDay'])['id'].size().reset_index()
a.columns=['Year_Week', 'WeekDay', 'Num_tickets']

a1=a.groupby(['Year_Week'])['WeekDay'].size().reset_index()
a2=a.groupby(['Year_Week'])[ 'Num_tickets'].mean().reset_index()
a3=pd.merge(a1, a2, how='left', on='Year_Week' )
a3=a3.loc[a3['WeekDay']==7]

tmp=pd.merge(tmp[['Year_Week', 'Num_orders']], a3[['Year_Week', 'Num_tickets']], how='left', on='Year_Week' )

tmp['TPR']=tmp['Num_tickets']/tmp['Num_orders']
n=2
tmp[f'Ord_{n}W'] = tmp['Num_orders'].rolling(window=n,min_periods=n).mean()
tmp[f'Tic_{n}W'] = tmp['Num_tickets'].rolling(window=n,min_periods=n).mean()
tmp[f'TPR_{n}W']=tmp[f'Tic_{n}W']/tmp[f'Ord_{n}W']

''' ---PLOT --------------------------------------'''

c1='Num_orders'
c2='Num_tickets'
color=plt.cm.tab20b(np.linspace(0,1,20))
fig, ax = plt.subplots(nrows=3, ncols=1,constrained_layout=True, figsize=(10,8))

ax[0].plot(tmp['Year_Week'], tmp[c1], color=color[4])
ax[0].plot(tmp['Year_Week'], tmp[f'Ord_{n}W'], color=color[6])
ax[0].legend(['Ord_1W',f'Ord_{n}W'], loc='upper left')

ax[1].plot(tmp['Year_Week'], tmp[c2],color=color[13] ) #color='mediumseagreen'
ax[1].plot(tmp['Year_Week'], tmp[f'Tic_{n}W'],color=color[15] ) #color='mediumseagreen'
ax[1].legend(['Tic_1W',f'Tic_{n}W'], loc='upper left')

ax[2].plot(tmp['Year_Week'], tmp['TPR'],color=color[0] ) #color='mediumseagreen'
ax[2].plot(tmp['Year_Week'], tmp[f'TPR_{n}W'],color=color[2] ) #color='mediumseagreen'

mean=tmp[f'TPR_{n}W'].mean()
ax[2].axhline(mean, color=color[15])
trans = transforms.blended_transform_factory(
    ax[2].get_yticklabels()[0].get_transform(), ax[2].transData)
ax[2].text(0,mean, "{:.3f}".format(mean), color=color[15], transform=trans, 
        ha="right", va="center")
ax[2].legend(['TPR',f'TPR_{n}W',f'TPR_{n}W_average'], loc='upper left')

plt.xlabel('Week', fontsize=12)
fig.suptitle('Tickets vs Orders (Aggregation Week)', fontsize=14)
# plt.xlim(xmin=0)

plt.show()

a1=df_tic.loc[df_tic['is_incoming']=='f']
a1['first_response_time'].mean()


a2=df_tic.loc[df_tic['is_incoming']=='t']
a2['first_response_time'].mean()

df_tic.groupby(['is_incoming'])['solve_respons_time'].mean()

''' -----------------------------------------'''
#moving average 7Days
c1='ymd'
a=df_ord.groupby([c1])['id'].size().reset_index()
a.columns=[c1,'Num_orders']

a1=df_tic.groupby([c1])['id'].size().reset_index()
a1.columns=[c1, 'Num_tickets']

tmp=pd.merge(a, a1, how='left', on=c1 ).sort_values(c1, ascending=True)

#plot
tmp['ymd']=pd.to_datetime(tmp['ymd'], infer_datetime_format=True)
color=plt.cm.tab20b(np.linspace(0,1,20))
fig, ax = plt.subplots(nrows=2, ncols=1,constrained_layout=False, figsize=(10,8))
ax[0].plot(tmp['ymd'], tmp['Num_orders'], color=color[4])
ax[0].axes.get_xaxis().set_visible(False)

ax[1].plot(tmp['ymd'], tmp['Num_tickets'], color=color[13]) 
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.xlabel('Date', fontsize=12)
fig.suptitle('Orders vs Tickets', fontsize=14)
plt.show()



tmp['Ord_7d'] = tmp['Num_orders'].rolling(window=7,min_periods=7).mean()
tmp['Tic_7d'] = tmp['Num_tickets'].rolling(window=7,min_periods=7).mean()
tmp['TPR_7d']=tmp['Tic_7d']/tmp['Ord_7d']

n=14
tmp[f'Ord_{n}d'] = tmp['Num_orders'].rolling(window=n,min_periods=n).mean()
tmp[f'Tic_{n}d'] = tmp['Num_tickets'].rolling(window=n,min_periods=n).mean()
tmp[f'TPR_{n}d']=tmp[f'Tic_{n}d']/tmp[f'Ord_{n}d']

print("Correlation between orders and tickets on a day basis is", round(tmp[['Num_orders','Num_tickets']].corr(),3  ).iloc[1,0])
''' ---PLOT --------------------------------------'''

fig, ax = plt.subplots(nrows=3, ncols=1,constrained_layout=False, figsize=(10,8))
# color=plt.cm.Pastel1(np.linspace(0,1,10))
color=plt.cm.tab20b(np.linspace(0,1,20))
# plt.set_ap("YlGnBu")
n1=7
ax[0].plot(tmp['ymd'], tmp[f'Ord_{n1}d'], color=color[4])
n2=14
ax[0].plot(tmp['ymd'], tmp[f'Ord_{n2}d'], color=color[5]) #, color=((33+n2)/255,99/255,170/255,1)

# ax[0].imshow(cmap='Pastel1')
ax[0].legend([f'Orders_{n1}d',f'Orders_{n2}d'], loc='upper left')
ax[0].axes.get_xaxis().set_visible(False)

ax[1].plot(tmp['ymd'], tmp[f'Tic_{n1}d'],color=color[13]) #color='mediumseagreen'
ax[1].plot(tmp['ymd'], tmp[f'Tic_{n2}d'], color=color[15]) #, color=((33+n2)/255,99/255,170/255,1)
ax[1].legend([f'Tickets_{n1}d',f'Tickets_{n2}d'], loc='upper left')
ax[1].axes.get_xaxis().set_visible(False)
# ax[1].set_xticks(rotation=90)
ax[2].plot(tmp['ymd'], tmp[f'TPR_{n1}d'],color=color[0]) #color='mediumseagreen'
ax[2].plot(tmp['ymd'], tmp[f'TPR_{n2}d'], color=color[2]) #, color=((33+n2)/255,99/255,170/255,1)
mean=tmp[f'TPR_{n2}d'].mean()
ax[2].axhline(mean, color=color[15])
trans = transforms.blended_transform_factory(
    ax[2].get_yticklabels()[0].get_transform(), ax[2].transData)
ax[2].text(0,mean, "{:.3f}".format(mean), color=color[15], transform=trans, 
        ha="right", va="center")
ax[2].legend([f'TPR_{n1}d',f'TPR_{n2}d', f'TPR_{n2}d_average' ], loc='upper left')

plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=90)
fig.suptitle('Tickets vs Orders (Moving Average)', fontsize=14)
plt.xlabel('Date', fontsize=12)
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.show()

'''----------------------------'''
# tickets per user type
c1='ymd'
a1=df_tic.groupby([c1,'user_type'])['id'].size().reset_index()
a1 =a1.pivot_table(index=c1,columns='user_type', values='id', aggfunc='sum').reset_index()
a1['ymd']=pd.to_datetime(a1['ymd'], infer_datetime_format=True)

#plot
plt.figure(figsize=(20, 7))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

for c in a1.columns.to_list():
    if c!='ymd':
        plt.plot(a1['ymd'], a1[c])
plt.legend([x for x in a1.columns.to_list() if x!='ymd'], loc='upper left')
plt.title('Number of tickets by user type')
plt.xticks(rotation=90)

plt.show()

'''----------------------------'''

#Filetered
c1='ymd'
a=df_ord.loc[df_ord['driver_id'].notnull()].groupby([c1])['id'].size().reset_index()
a.columns=[c1,'Num_orders']

a1=df_tic.groupby([c1])['id'].size().reset_index()
a1.columns=[c1, 'Num_tickets']

tmp=pd.merge(a, a1, how='left', on=c1 ).sort_values(c1, ascending=True)

#plot
tmp['ymd']=pd.to_datetime(tmp['ymd'], infer_datetime_format=True)
color=plt.cm.tab20b(np.linspace(0,1,20))
fig, ax = plt.subplots(nrows=2, ncols=1,constrained_layout=False, figsize=(10,8))
ax[0].plot(tmp['ymd'], tmp['Num_orders'], color=color[4])
ax[0].axes.get_xaxis().set_visible(False)

ax[1].plot(tmp['ymd'], tmp['Num_tickets'], color=color[13]) 
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.xlabel('Date', fontsize=12)
fig.suptitle('Orders vs Tickets', fontsize=14)
plt.show()



tmp['Ord_7d'] = tmp['Num_orders'].rolling(window=7,min_periods=7).mean()
tmp['Tic_7d'] = tmp['Num_tickets'].rolling(window=7,min_periods=7).mean()
tmp['TPR_7d']=tmp['Tic_7d']/tmp['Ord_7d']

n=14
tmp[f'Ord_{n}d'] = tmp['Num_orders'].rolling(window=n,min_periods=n).mean()
tmp[f'Tic_{n}d'] = tmp['Num_tickets'].rolling(window=n,min_periods=n).mean()
tmp[f'TPR_{n}d']=tmp[f'Tic_{n}d']/tmp[f'Ord_{n}d']

print("Correlation between orders and tickets on a day basis is", round(tmp[['Num_orders','Num_tickets']].corr(),3  ).iloc[1,0])
''' ---PLOT --------------------------------------'''

fig, ax = plt.subplots(nrows=3, ncols=1,constrained_layout=False, figsize=(10,8))
# color=plt.cm.Pastel1(np.linspace(0,1,10))
color=plt.cm.tab20b(np.linspace(0,1,20))
# plt.set_ap("YlGnBu")
n1=7
ax[0].plot(tmp['ymd'], tmp[f'Ord_{n1}d'], color=color[4])
n2=14
ax[0].plot(tmp['ymd'], tmp[f'Ord_{n2}d'], color=color[5]) #, color=((33+n2)/255,99/255,170/255,1)

# ax[0].imshow(cmap='Pastel1')
ax[0].legend([f'Orders_{n1}d',f'Orders_{n2}d'], loc='upper left')
ax[0].axes.get_xaxis().set_visible(False)

ax[1].plot(tmp['ymd'], tmp[f'Tic_{n1}d'],color=color[13]) #color='mediumseagreen'
ax[1].plot(tmp['ymd'], tmp[f'Tic_{n2}d'], color=color[15]) #, color=((33+n2)/255,99/255,170/255,1)
ax[1].legend([f'Tickets_{n1}d',f'Tickets_{n2}d'], loc='upper left')
ax[1].axes.get_xaxis().set_visible(False)
# ax[1].set_xticks(rotation=90)
ax[2].plot(tmp['ymd'], tmp[f'TPR_{n1}d'],color=color[0]) #color='mediumseagreen'
ax[2].plot(tmp['ymd'], tmp[f'TPR_{n2}d'], color=color[2]) #, color=((33+n2)/255,99/255,170/255,1)
mean=tmp[f'TPR_{n2}d'].mean()
ax[2].axhline(mean, color=color[15])
trans = transforms.blended_transform_factory(
    ax[2].get_yticklabels()[0].get_transform(), ax[2].transData)
ax[2].text(0,mean, "{:.3f}".format(mean), color=color[15], transform=trans, 
        ha="right", va="center")
ax[2].legend([f'TPR_{n1}d',f'TPR_{n2}d', f'TPR_{n2}d_average' ], loc='upper left')

plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=90)
fig.suptitle('Tickets vs Orders (Moving Average)', fontsize=14)
plt.xlabel('Date', fontsize=12)
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.show()






'''-----------------------------------------------'''


# print('Negative values\n',df.iloc[:,1:].lt(0).sum().sum())

# print('Max values\n',df.max())

# df.fillna(0, inplace=True)
# print('Missing (null) values\n',df.isnull().sum())


'''-----------------------------------------------'''
'''-----------------------------------------------'''

#0 EDA orders WeekDay and Hour distribution
nom=1000
c1='WeekDay'
c2='Hour'
c3='id'

def hit_map(df, c1, c2, c3, agr='size', lb='', nm=1):
    df_gr=df.groupby([c1,c2])[[c3]].agg(agr).reset_index()
    df_gr.columns=[c1,c2,c3]
    df_gr[c3]/=nm
    df_gr = df_gr.pivot_table(index=c2,columns=c1, values=c3, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df_gr, linewidths=.5, annot=True, ax=ax, cmap = "YlGnBu")
    ax.set_xticklabels(day_names)
    ax.set_title(f'{lb} "{c3}" {agr} (/{nm})') 
    plt.show()



hit_map(df_ord, c1, c2, c3, 'size', 'Orders', nom)
hit_map(df_tic, c1, c2, c3, 'size',  'Tickets', nom)

#completed
hit_map(df_ord.loc[df_ord['completed']=='t'], c1, c2, c3, 'size', 'Orders completed', nom)

hit_map(df_tic, c1, c2, 'total_resolution_time', 'sum', 'Time to solve', 1)
df_gr=df_tic.groupby([c1,c2])[['total_resolution_time']].agg('sum').reset_index()
df_tic.columns

sns.distplot(df_tic['total_resolution_time'], kde=True, rug=False)
sns.rugplot(df_tic['total_resolution_time'])

c1='first_response_time'
df_tic[c1].describe()
#plot distribution
ax=sns.distplot(df_tic[c1], kde=False, rug=True,hist_kws={'log':True,"color":color[0]},  rug_kws={"color":color[1]})
ax.set_ylabel("Tickets")
ax.set_xlabel("Hours")
plt.title(f'{c1} (in log scale)')
plt.show()


'''-----------------------------------------------'''
c1='solve_respons_time'


#plot distribution
ax=sns.distplot(df_tic[c1], kde=False, rug=True,hist_kws={'log':True,"color":color[0]},  rug_kws={"color":color[1]})
ax.set_ylabel("Tickets")
ax.set_xlabel("Hours")
plt.title(f'{c1} (in log scale)')
plt.show()


df_tic.loc[df_tic[c1]==0].shape[0]/df_tic.shape[0]
df_tic.loc[(df_tic[c1]>0) & (df_tic[c1]<=8)].shape[0]/df_tic.shape[0]

#tickets per hour
nm=int(1/df_tic.loc[(df_tic[c1]>=0) & (df_tic[c1]<=1), c1].mean())

c1='WeekDay'
c2='Hour'
c3='id'


# Shedule for customer support
df_gr=df_tic.groupby([c1,c2])[[c3]].agg('size').reset_index()
df_gr.columns=[c1,c2,c3]
df_gr[c3]=round(df_gr[c3]/nm,0)
df_gr = df_gr.pivot_table(index=c2,columns=c1, values=c3, aggfunc='mean')
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df_gr, linewidths=.5, annot=True, ax=ax, cmap = "YlGnBu")
ax.set_xticklabels(day_names)
ax.set_title(f'People needed ({nm} tickets per hour)') 
plt.show()    













