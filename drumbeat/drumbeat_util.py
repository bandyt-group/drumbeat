import numpy as np
import sys
import networkx as nx


## Input and Output Functions


## Plotting Functions for tracks and WDegree ##

def plotTM(ax,time=None,tm=None,color='grey'):
    axy=ax.twinx()
    if time is None:
        axy.plot(tm,color=color,alpha=0.75)
        axy.tick_params(axis='y',labelsize=20,labelcolor=color)
        return axy
    axy.plot(time,tm,color=color,alpha=0.75)
    axy.tick_params(axis='y',labelsize=20,labelcolor=color)
    return axy

def plotwd(ax,trbn,nodestoplot,colors=None):
    if colors is not None:
        [ax.plot(w,linewidth=4,color=colors[i]) for i,w in enumerate(trbn.wdegree[np.in1d(trbn.nodes,nodestoplot)])] 
        return
    [ax.plot(i,linewidth=4) for i in trbn.wdegree[np.in1d(trbn.nodes,nodestoplot)]]
    



def maxargstype(maxargs,iv,interval):
    return np.where((maxargs[iv]>interval[0])&(maxargs[iv]<=interval[1]))[0]

def getednums(edgenames,edge):
    return np.array([i for i,j in enumerate(edgenames) if edge in j])

def getnodes(edges,unique=True):
    if unique==True:
        return np.unique(np.concatenate(np.array([i.split('->') for i in edges])))
    return np.concatenate(np.array([i.split('->') for i in edges]))


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
def getedgesinpeak(trbn,contact,time,thresh=None,returnvalues=False):
    contedges=np.array([ed for ed in trbn.edges if contact in ed])
    if thresh is None:
        thresh=trbn.tracks[np.in1d(trbn.edges,contedges)][:,time].mean()
        print('Mean:',thresh)
    edges=trbn.edges[np.in1d(trbn.edges,contedges)][np.where(trbn.tracks[np.in1d(trbn.edges,contedges)][:,time]>thresh)[0]]
    if returnvalues:
        return np.column_stack((edges,trbn.tracks[np.in1d(trbn.edges,edges)][:,time]))
    return edges

# Creating network table

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


