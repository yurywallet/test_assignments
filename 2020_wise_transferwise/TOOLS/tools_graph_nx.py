# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:47:33 2020

@author: Yury
"""
import networkx as nx
import pandas as pd

def link_graph(dff):


    dff.columns=['a','b', 'v']
    G = nx.DiGraph()
    
    edges=list(zip(dff['a'],dff['b'], dff['v']))
    
    G.add_weighted_edges_from(edges, weight='weight')
    
    # G.out_degree()
    # G.out_degree(weight='weight')
    a=dict(G.in_degree())
    dd=pd.DataFrame()
    dd['user_id']=a.keys()
    dd['links_in']=a.values()
    a=dict(G.in_degree(weight='weight'))
    dd['links_in_val']=a.values()
    
    
    b=dict(G.out_degree())
    dd2=pd.DataFrame()
    dd2['user_id']=b.keys()
    dd2['links_out']=b.values()
    b=dict(G.out_degree(weight='weight'))
    dd2['links_out_val']=b.values()
    
    ab=pd.merge(dd,dd2, on='user_id', how='left')
    ab['links_num']=ab['links_in']+ab['links_out']
    ab['links_val']=ab['links_in_val']-ab['links_out_val']
    

    # ab['clust_coef']=[nx.clustering(G, x) for x in G.nodes]
    # print (nx.average_clustering(G))
    
    
    return ab


def plot_digraph(dff, weight=True, pixels=5000,dpi=200, remap=False):
    import matplotlib.pyplot as plt
    
    import networkx as nx
    try:
        # conda install -c anaconda pydot
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
        layout = graphviz_layout
            
    except ImportError:
        try:
            # conda install -c alubbock pygraphviz
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout
            layout = graphviz_layout
        except ImportError:
            print("PyGraphviz and pydot not found;\n"
                  "drawing with spring layout;\n"
                  "will be slow.")
            layout = nx.spring_layout
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    dff.columns=['a','b', 'v']
    G = nx.DiGraph()
    
    # weight=False
    
    if weight:
        edges=list(zip(dff['a'],dff['b'], dff['v']))
        G.add_weighted_edges_from(edges, weight='weight')
    else:
        edges=list(zip(dff['a'],dff['b']))
        G.add_edges_from(edges)
    
    #relabel
    a=dict(G.degree())
    dd=pd.DataFrame()
    dd['user_id']=a.keys()
    dd['links_in']=a.values()
    
    mapping={}
    if remap:
        mapping=dict(zip(dd['user_id'],dd.index))
        G = nx.relabel_nodes(G, mapping, copy=False)
    
    # for i in range(dff.shape[0]):
    #     #, weight=1)
    #     G.add_edge(dff.iloc[i]['a'],dff.iloc[i]['b'], color='red', weight=dff.iloc[i]['v'])
    
    # G.nodes
    # node_color = [G.degree(v) for v in G]
    
    max_out=max(dict(G.out_degree()).values())
    max_in=max(dict(G.in_degree()).values())
    
    min_out=min(dict(G.out_degree()).values())
    min_in=min(dict(G.in_degree()).values())

    cmap = cm.rainbow
    norm_in = Normalize(vmin=min_in, vmax=max_in)
    norm_out = Normalize(vmin=min_out, vmax=max_out)
    # node_out = [cmap(norm_out(G.out_degree(v))) for v in G]
    node_out = [500.0 * G.out_degree(v) for v in G.nodes()]
    
    
    # node_in = [cmap(norm_in(G.in_degree(v) )) for v in G]
    node_in = [50.0*G.in_degree(v) for v in G.nodes()]
    
    node_size= [500.0 * G.degree(v) for v in G.nodes()]
    
    carrow=[cmap(norm_out(G.out_degree(v))) for v in G.nodes()]
    
    # betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
    # node_size =  [v * 10000 for v in betCent.values()]
    
    
    if weight:
        # norm_width = Normalize(vmin=dff['v'].min(), vmax=dff['v'].max())
        # weights = [norm_width(G[u][v]['weight']) for u,v in G.edges]
        
        import numpy as np
        def log_norm_weight(x, mi, eps=0.1):
            log_mi=np.log(eps)
            log_x=np.log(x-mi+eps)-log_mi
            return log_x
        # weights = [log_norm_weight(G[u][v]['weight'], mi=dff['v'].min(), eps=0.1) for u,v in G.edges]        
    
        
        def norm_weight(x, mi, ma):
            x_new=(x-mi)/ma
            return x_new

         
        def std_norm_weight(x, mi, mean, stdv):
            x_new=(x-mean)/stdv-(mi-mean)/stdv+1
            return x_new
        weights = [std_norm_weight(G[u][v]['weight'], mi=dff['v'].min(), mean=dff['v'].mean(), stdv=dff['v'].std()) for u,v in G.edges]     
        
       
    


    

    #plot----------------------------    
    # 5000 X5000
    w=pixels
    h=w
    plt.figure(figsize=(4+w/dpi,4+h/dpi), dpi=dpi)
    
    
    
    # nx.spring_layout(G)
    if weight:
        nx.draw_networkx(G,layout(G), 
                           # font_color="grey",
                          # with_labels=False  ,
                          alpha=0.8,
                          node_color=node_out, 
                          node_size=node_size,
                          # linewidths=node_in,
                          edge_color=carrow,
                          width=weights)
    else:
          nx.draw_networkx(G,layout(G), 
                           # font_color="grey",
                          # with_labels=False  ,
                          alpha=0.8,
                          node_color=node_out, 
                          node_size=node_size,
                          # linewidths=node_in,
                          edge_color=carrow
                          # ,
                          # edge_color=node_in
                            # edge_vmin,
                            # edge_vmax
                          # edge_cmap=plt.cm.rainbow
                          )
           
    # print("In-degree for all nodes: ", dict(G.in_degree())) 
    # print("Out degree for all nodes: ", dict(G.out_degree)) 
    
    return mapping


def find_all_paths_new2(dff):
    print('Creating paths...')
    import networkx as nx
    dff.columns=['a','b']
    edges=list(zip(dff['a'],dff['b']))
    edg=[[x[0],x[1]] for x in edges]
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    a=dict(G.in_degree())
    dd=pd.DataFrame()
    dd['user_id']=a.keys()
    dd['links_in']=a.values()
        
    b=dict(G.out_degree())
    dd2=pd.DataFrame()
    dd2['user_id']=b.keys()
    dd2['links_out']=b.values()
    
    ab=pd.merge(dd,dd2, on='user_id', how='left')
    
    
    nodes_send=ab.loc[ab['links_out']>0,'user_id']
    nodes_rec=ab.loc[ab['links_in']>0,'user_id']
    
    print (' start ', len(nodes_send), '| end ' , len (nodes_rec))


    paths=edg.copy()
    # import time
    # m=0
    # tt=0
    for i in nodes_send:
        for j in nodes_rec:
            # if m<10000:
                # print(1)
            if i!=j and [i,j] not in edg:
                # t=time.time()
                paths+=list(nx.all_simple_paths(G, i,j))
                # tt+=time.time()-t
                # m+=1
            # else: break
                
    return G, paths   


def find_long_paths_new(dff):
    print('Creating paths...')
    import time
    t_start=time.time()
    
    
    import networkx as nx
    dff.columns=['a','b']
    edges=list(zip(dff['a'],dff['b']))
    edg=[[x[0],x[1]] for x in edges]
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    a=dict(G.in_degree())
    dd=pd.DataFrame()
    dd['user_id']=a.keys()
    dd['links_in']=a.values()
        
    b=dict(G.out_degree())
    dd2=pd.DataFrame()
    dd2['user_id']=b.keys()
    dd2['links_out']=b.values()
    
    ab=pd.merge(dd,dd2, on='user_id', how='left')
    
    # nodes_send=ab.loc[(df_links['links_out']>0) & (ab['links_in']==0),'user_id']
    # nodes_rec=ab.loc[(ab['links_in']>0) & (ab['links_out']==0),'user_id']
    
    nodes_send=ab.loc[(df_links['links_out']>0) ,'user_id']
    nodes_rec=ab.loc[(ab['links_in']>0),'user_id']
    
    # nodes_midl=ab.loc[(ab['links_in']>0) & (ab['links_out']>0),'user_id']

    print (' start ', len(nodes_send), '| end ' , len (nodes_rec))
    t_num=len(nodes_send)*len (nodes_rec)

    paths=edg.copy()
    # import time
    # m=0
    l=0
    # tt=0
    # ttt=0
    
    for i in nodes_send:
        for j in nodes_rec:
            l+=1
            # if m<10000:
                # print(1)
            #ONLY if a pair is not an EDGE
            if i!=j and [i,j] not in edg:
                # t=time.time()
                paths+=list(nx.all_simple_paths(G, i,j))
            #         tt+=time.time()-t
            #         m+=1
            
            if l%10000==0: 
                t_cur=time.time()-t_start
                t_est=t_cur/(100*l/t_num)/60/60
                print (f'Done: {round(100*l/t_num,4)} %  | Time from start: {round(t_cur,3)} sec ({round(t_cur/60,2)} min) | Estimated {round(t_est,1)} hours ')
            
            # else: break
            

    
    # print( m ,l )        
    return G, paths 

# G, all_paths=find_long_paths_new(df[['user_id', 'target_recipient_id']])


# 73000*69000*tt/79000/60/60/24




def find_max_path(A):
    print('MAX paths...')
    G=A[0]
    paths=A[1]
    df_path=pd.DataFrame(columns=['user_id','max_path_len','max_path'])
    for i in list(G.nodes()):
        
        l= [len(x) for x in paths if x[0]==i]
        if len(l)>0:
            ll=max(l)
            l=[x for x in paths if x[0]==i and len(x)==ll]
        else:
            ll=1
            l=[]
        df_path.loc[df_path.shape[0]]=[i,ll-1,l]
    return df_path

# df_paths=find_max_path(find_long_paths_new(dff[['a','b']]))
 
# df_paths=find_max_path(find_long_paths_new(df[['user_id', 'target_recipient_id']]))




def connected_nodes(dff):    
    import networkx as nx
    dff.columns=['a','b', 'v']
    G = nx.Graph() 
    
    edges=list(zip(dff['a'],dff['b'], dff['v']))
    G.add_weighted_edges_from(edges, weight='weight')
    # G = nx.Graph()
    # G.add_edges_from(edges)
    print("Number of subgraphs", nx.number_connected_components(G)) 
    # print(list(nx.connected_components(G))) 
    l=list(nx.connected_components(G))
    ll=[len(x) for x in l]
    df_temp=pd.DataFrame(columns=['F_groups', 'F_groups_size'])
    df_temp['F_groups']=l
    df_temp['F_groups_size']=ll
    df_temp.sort_values(['F_groups_size'], ascending=False, inplace=True)
    
    df_temp['F_groups_turnover']=df_temp['F_groups'].apply(lambda x: dff.loc[dff['a'].isin(x), 'v'].sum())
    return df_temp
