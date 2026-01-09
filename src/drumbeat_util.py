import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import drumbeat as db

## Input and Output Functions


### Top nodes ###

def gettopnodes(D,top=20,cutoff=0.8):
    u,c=u,c=np.unique_counts(np.concatenate([d.nodes[d.wdsort][:top] for d in D]))
    return u[np.flip(np.argsort(c))][c[np.flip(np.argsort(c))]>=cutoff*len(D)]

def getedges(D,nodes):
    return np.unique(np.concatenate([[i for i in D[0].edges if j in i] for j in nodes]))

def nodesfromeds(edges):
    return np.unique(np.concatenate([i.split('->') for i in edges]))


### PCA analysis ###

def subset_pca(D,MD,top_nodes=None):
    if top_nodes is not None:
        nodes=nodesfromeds(getedges(D,top_nodes))
    if top_nodes is None:
        nodes=nodesfromeds(getedges(D,gettopnodes(D,top=topnodes)))
    print(f'Building PCA using {nodes.size} nodes')
    Tc=db.getuniversaldataset(MD,concat=True,union=True)
    Tall=Tc.traj[:,np.isin(Tc.labels,nodes)].astype(int)
    Tmean=Tall-Tall.mean(0)
    u,s,v=np.linalg.svd(Tmean,False)
    print(f'PCA built using {nodes.size} nodes and {u.shape[0]} frames')
    return u,s,v


def plot_pca_contour(u,pcs=[0,1]):
    import seaborn as sns
    g=sns.jointplot(
       x=u[:,pcs[0]],
       y=u[:,pcs[1]],
       kind="kde",        # contour density plot
       fill=True,         # fill contours
       cmap="Blues"
    )
    return g
## Plotting Functions for tracks and WDegree ##

def trajectory_index_ranges(MD):
    ranges = []
    start = 0

    for md in MD:
        n_frames = md.traj.shape[0]
        end = start + n_frames
        ranges.append((start, end))
        start = end

    return np.array(ranges)


def plotTM(ax,time=None,tm=None,color='grey'):
    axy=ax.twinx()
    if time is None:
        axy.plot(tm,color=color,alpha=0.75)
        axy.tick_params(axis='y',labelsize=20,labelcolor=color)
        return axy
    axy.plot(time,tm,color=color,alpha=0.75)
    axy.tick_params(axis='y',labelsize=20,labelcolor=color)
    return axy

def plotwd(D,nodestoplot,colors=None):
    fig,ax=plt.subplots(1,1,figsize=(16,9))
    if colors is not None:
        [ax.plot(w,linewidth=4,color=colors[i]) for i,w in enumerate(D.wdegree[np.isin(D.nodes,nodestoplot)])] 
        return
    [ax.plot(i,linewidth=4) for i in D.wdegree[np.isin(D.nodes,nodestoplot)]]
    ax.legend(D.nodes[np.isin(D.nodes,nodestoplot)],fontsize=24)
    return fig,ax    

def maxargstype(maxargs,iv,interval):
    return np.where((maxargs[iv]>interval[0])&(maxargs[iv]<=interval[1]))[0]

def getednums(edgenames,edge):
    return np.array([i for i,j in enumerate(edgenames) if edge in j])
## Peaks functions ##

def findpeaksintrac(trac,thresh=0.2,distance=200):
    return find_peaks(trac,height=thresh,distance=200)

def peaksinrange(intv,WD,thresh=0.2,inclusive='all'):
    Tb=WD>thresh
    # 'only' means the nodes WD peaks above thresh only in the interval and nowhere
    #   else
    if inclusive=='only':
        Edb=np.array([(np.any(Tb[i,intv[0]:intv[1]]))&(np.all(~Tb[i,:intv[0]]))&(np.all(~Tb[i,intv[1]:])) for i in range(Tb.shape[0])])
        return Edb
    # 'all' means peaks in interval and can also peak anywhere else
    if inclusive=='all':
        Edb=np.array([np.any(Tb[i,intv[0]:intv[1]]) for i in range(Tb.shape[0])])
        return Edb
    # 'before' means peaks in the interval as well as before the range
    if inclusive=='before':
        Edb=np.array([(np.any(Tb[i,intv[0]:intv[1]]))&(np.all(~Tb[i,intv[1]:])) for i in range(Tb.shape[0])])
        return Edb
    # 'after' means peaks in the interval as well as after
    if inclusive=='after':
        Edb=np.array([(np.any(Tb[i,intv[0]:intv[1]]))&(np.all(~Tb[i,:intv[0]])) for i in range(Tb.shape[0])])
        return Edb


# edges within a peak
def getedgesinpeak(trbn,time,contact=None,edgelist=None,thresh=None,returnvalues=False):
    if edgelist is None:
        edgelist=np.array([ed for ed in trbn.edges if contact in ed])
    if thresh is None:
        thresh=trbn.tracks[np.in1d(trbn.edges,edgelist)][:,time].mean()
        print('Mean:',thresh)
    edges=trbn.edges[np.in1d(trbn.edges,edgelist)][np.where(trbn.tracks[np.in1d(trbn.edges,edgelist)][:,time]>thresh)[0]]
    if returnvalues:
        return np.column_stack((np.array([i.split('->') for i in edges]),trbn.tracks[np.in1d(trbn.edges,edges)][:,time]))
    return edges


# Edges in all peaks ##

def alledgesandindx(D,peaktimes,edgelist,threshold=0.05):
    alled=np.unique(np.concatenate([getedgesinpeak(D,time=p,edgelist=edgelist,thresh=threshold) for p in peaktimes]))
    edind=np.array([np.where(D.edges==ed)[0][0] for ed in alled])
    return alled, edind


# Creating network table

def createtable(D,alled,edind,peaks):
    data=np.column_stack((np.array([i.split('->') for i in alled]),np.round(D.tracks[edind][:,peaks],4)))

def getnodepeaks(Ds,nodes,fixzero=True):
    X=np.array([[np.argmax(Ds[i].wdegree[np.where(Ds[i].nodes==j)])for i in range(len(Ds))] for j in nodes])
    if fixzero:
        for i in range(len(nodes)):
            X[i][X[i]<50]=50
        return X
    return X

def getalledges(Ds,nodestoplot,nodepeaks,thresh=0.2):
    numDs=len(Ds)
    numnodes=len(nodestoplot)
    return np.concatenate([np.concatenate([getedgesinpeak(Ds[i],nodestoplot[j],nodepeaks[j][i],thresh=thresh,returnvalues=True) for i in range(numDs)]) for j in range(numnodes)])

def createtabledic(alledges):
    tabledic=dict()
    for k,v in alledges:
        if k not in tabledic.keys():
            tabledic[k]=v
        if v>tabledic[k]:
            tabledic[k]=v
    return tabledic

def converttonetworktable(edsandvals):
    return np.column_stack((np.array([i[0].split('->') for i in edsandvals]),edsandvals[:,1]))

def dicttonetworktable(tabledic):
    return np.vstack((['source','target','weight'],converttonetworktable(np.column_stack((list(tabledic.keys()),list(tabledic.values()))))))

def graph_from_edge_array(edge_array):
    G = nx.Graph()
    for src, tgt, w in edge_array:
        G.add_edge(src, tgt, weight=float(w))
    return G



# degree and betweenness
def initgraph(nodes):
    G=nx.Graph()
    G.add_nodes_from(nodes)
    return G

def getevals(T,t):
    evals=np.copy(T[:,t])
    return evals

def addedges(G,edges,evals,thresh):
    G.add_edges_from(edges[evals>thresh])
    return G

def clear_edges(G):
    return nx.create_empty_copy(G)

def get_Degree(G):
    return np.array(list(nx.degree_centrality(G).values()))

def get_Between(G):
    return np.array(list(nx.betweenness_centrality(G).values()))

class Scan():
    def __init__(self,edges,nodes,T,thresh=0.01):
        self.edges=edges
        self.nodes=nodes
        self.T=T
        self.Tmax=T.shape[1]
        self.thresh=thresh

        #Make windows
        x=np.arange(0,T.shape[1],300)
        y=x+300
        self.W=np.column_stack((x,y))[:-1]

    def deg_bet_t(self,t):
        iG=initgraph(self.nodes)
        self.G=addedges(iG,self.edges,getevals(self.T,t),self.thresh)
        D=get_Degree(self.G)
        B=get_Between(self.G)
        return np.array([D,B])
    
    def deg_bet_deltat(self,window):
        return np.array([self.deg_bet_t(t) for t in np.arange(window[0],window[1])])

def getnodedict(nodes):
    nodedict={}
    for i,n in enumerate(nodes):
        nodedict[n]=i
    return nodedict

def splitedge_indices(nodedict,edgenames):
    return np.array([[nodedict[np.array([i.split('->') for i in edgenames])[i][j]] for j in range(2)] for i in range(edgenames.shape[0])])
    
    

# Given TM3-TM6 Distances, compute SMA
def computesma(TMs):
    return [np.convolve(tm,np.ones(500),'valid')/500 for tm in TMs]


def network_entropy(weights, eps=1e-12):
    """
    Compute Shannon entropy from a list/array of edge weights.
    Parameters
    ----------
    weights : array-like
        Edge weights (e.g. MI values).
    eps : float
        Small value to avoid log(0).
    """
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]  # drop zero-weight edges
    if len(w) == 0:
        return 0.0
    p = w / (w.sum() + eps)
    return -np.sum(p * np.log(p + eps))


def node_entropy(G, weight='weight', eps=1e-12):
    # weighted degrees
    degrees = np.array([
        d for _, d in G.degree(weight=weight)
    ], dtype=float)
    degrees = degrees[degrees > 0]
    if len(degrees) == 0:
        return 0.0
    p = degrees / (degrees.sum() + eps)
    return -np.sum(p * np.log(p + eps))

def plot_weighted_graph(
    G,
    node_size=800,
    edge_scale=5.0,
    with_labels=True,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=1)
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    widths = edge_scale * weights / weights.max()
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color="lightgray",
        edgecolors="black",
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        width=widths,
        alpha=0.8,
        ax=ax
    )
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    ax.set_axis_off()

def plot_subnetworks_over_time(edge_arrays, timepoints):
    n = len(timepoints)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, timepoints):
        G = dutil.graph_from_edge_array(edge_arrays[t])
        plot_weighted_graph(G, ax=ax)
        ax.set_title(f"t = {t}")
    plt.tight_layout()


## Plot B2AR WD
#trajs=np.arange(14)
#axys=[a.twinx() for a in axs]
#[axys[n].plot(b2_T[i],TM3_B[i],'black',alpha=0.5) for n,i in enumerate(trajs)]
#[axys[n].plot(b2_T[i][-SMA[i].shape[0]:],SMA[i],'black') for n,i in enumerate(trajs)]
#[[axs[n].plot(b2_T[j][:-1],y,color='brown',alpha=0.3,linewidth=1) for y in B[j].wdegree[np.in1d(B[j].nodes,top50)]] for n,j in enumerate(trajs)]
#[[axs[n].plot(b2_T[j][:-1],y,color=Col[i],linewidth=3) for i,y in enumerate(B[j].wdegree[np.in1d(B[j].nodes,Bnodes)][b2_indx])] for n,j in enumerate(trajs)]
#axs[2].legend(handles=axs[2].lines[-7:],labels=nodesb3_bw,fontsize=18,loc=1)


## Plot single trajctory
#axys=axs.twinx()
#i=9
#axys.plot(b2_T[i],TM3_B[i],'black',alpha=0.5)
#axys.plot(b2_T[i][-SMA[i].shape[0]:],SMA[i],'black')
#[axs.plot(b2_T[i][:-1],y,color='brown',alpha=0.3,linewidth=1) for y in B[i].wdegree[np.in1d(B[i].nodes,top50)]]
#[axs.plot(b2_T[i][:-1],y,color=Col[j],linewidth=5) for j,y in enumerate(B[i].wdegree[np.in1d(B[i].nodes,Bnodes)][b2_indx])]
#axs.legend(handles=axs[2].lines[-7:],labels=nodesb3_bw,fontsize=18,loc=1)




