import numpy as np
import os
import sys
import itertools
import multiprocessing
import csv
import pickle
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from networkx.algorithms.moral import moral_graph
import shortenRes as sR
import tsvloader


## File input/output ##

def csvreader(inputfile):
    out = []
    with open(inputfile, newline = '') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            out.append(row)
    return np.array(out)

def csvwrite(file,data,labels=None):
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if labels is not None:
            csv_writer.writerow(labels)
        for row in data:
            csv_writer.writerow(row)

def picklewrite(output,data):
    with open(output, 'wb') as file:
        pickle.dump(data, file)

def pickleread(picklefile):
    with open(picklefile, 'rb') as file:
        return pickle.load(file)


## Parrallel function ##

def runParallel(foo,iter,ncore):
    pool=multiprocessing.Pool(processes=ncore)
    try:
        out=(pool.map_async( foo,iter )).get()
    except KeyboardInterrupt:
        print ("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        #print ("Quitting normally core used ",ncore)
        pool.close()
        pool.join()
    try:
        return out
    except Exception:
        return out


## Entropy and Mutual Information Functions ##

def encode(c):
    try:
        b=np.ones(c.shape[1],dtype=int)
    except Exception:
        c=np.column_stack(c)
        b=np.ones(c.shape[1],dtype=int)
    b[:-1]=np.cumprod((c[:,1:].max(0)+1)[::-1])[::-1]
    return np.sum(b*c,1)


def mi_p(A):
    X,Y=A
    return H(X)+H(Y)-joinH((X,Y))

def H(i):
        """entropy of labels"""
        p=np.unique(i,return_counts=True)[1]/i.size
        return -np.sum(p*np.log2(p))

def joinH(i):
    pair=np.column_stack((i))
    en=encode(pair)
    p=np.unique(en,return_counts=True)[1]/len(en)
    return -np.sum(p*np.log2(p))



## Trajectory class operations ##

def transformRes(pair):
    R=sR.Residue()
    res1=R.shRes(pair.split('_')[0][:3])+pair.split('_')[0][3:]
    res2=R.shRes(pair.split('_')[1][:3])+pair.split('_')[1][3:]
    return res1+'_'+res2

def transformResidues(labels,traj):
    straj=traj[:,np.argsort(np.array([transformRes(l) for l in labels]))]
    slabels=np.sort(np.array([transformRes(l) for l in labels]))
    return slabels,straj

def remove_Neighbors(contacts,N):
    iplus=np.array([i for i,p in enumerate(contacts) if int(p.split('_')[0][1:])+N !=
 int(p.split('_')[1][1:])])
    imin=np.array([i for i,p in enumerate(contacts) if int(p.split('_')[0][1:])-N !=
int(p.split('_')[1][1:])])
    return np.intersect1d(iplus,imin)

def getMImatrix(traj,numproc):
    MI=runParallel(mi_p,itertools.combinations(traj,2),numproc)
    D=np.zeros((traj.shape[0],traj.shape[0]))
    D[np.triu_indices_from(D,1)]=MI
    return D+D.T

def applyMIfilter(labels,traj,MImatrix=None,th=0.005,nproc=4):
    if MImatrix is None:
        MImatrix=getMImatrix(traj.T,nproc)
    indx_pair2remove=np.all(MImatrix<th,1)
    return labels[~indx_pair2remove],traj[:,~indx_pair2remove],MImatrix


## Main Trajectory class ##

class Traj():
    
    def __init__(self,labels,traj):
        self.input_traj=traj[:,np.argsort(labels)]
        self.input_labels=labels[np.argsort(labels)]
        self.traj=self.input_traj
        self.labels=self.input_labels   
        self.MI=None
        
    def restore_input_traj(self):
        self.traj=self.input_traj
        self.labels=self.input_labels
    
    def cuttraj(self,start,end):
        self.traj=self.traj[:,start:end]

    def subsettraj(self,interval):
        self.traj=self.traj[::interval]

    def transformResidues(self):
        self.labels,self.traj=transformResidues(self.labels,self.traj)      

    def resandneigh(self):
        self.transformResidues()
        self.remove_Neighbor(N=1)    

    def remove_singles(self):
        self.iv=[i for i,x in enumerate(self.traj.T) if len(set(x))>1]
        self.traj=self.traj[:,self.iv]
        self.labels=self.labels[self.iv]

    def remove_Neighbor(self,N):
        I=remove_Neighbors(self.labels,N)
        self.labels=self.labels[I]
        self.traj=self.traj[:,I]
        
    def compute_MI_matrix(self,numproc=4):
        self.MI=getMImatrix(self.traj,numproc)
    
    def MIfeatureselect(self,th,numproc=4):
        if self.MI is None:
            self.labels,self.traj,self.MI=applyMIfilter(self.labels,self.traj,MImatrix=self.MI,th=th,nproc=numproc)
            return
        self.restore_input_traj()
        self.transformResidues()
        self.remove_Neighbor(N=1)
        self.labels,self.traj,self.MI=applyMIfilter(self.labels,self.traj,MImatrix=self.MI,th=th,nproc=numproc)

## Trajectory Data Loader from files ##

def gettrajfromtsv(file,tmax=None):
    return tsvloader.gettrajfromtsv(file)    

def gettrajfromcsv(file):
    out=csvreader(file)
    labels=out[0]
    traj=out[1:].astype(bool)
    return Traj(labels,traj)

def loadtrajensemble(files):
    if files[0][-3:]=='tsv':
        T=[gettrajfromtsv(f) for f in files]
        Ts=[Traj(labels=t[0],traj=t[1]) for t in T]
        [t.resandneigh() for t in Ts]
        return Ts
    if files[0][-3:]=='csv': 
        T=[gettrajfromcsv(f) for f in files]
        Ts=[Traj(labels=t[0],traj=t[1]) for t in T]
        [t.resandneigh() for t in Ts]
        return Ts
    print('File Type not recognized: Expecting .tsv or .csv')

## Sampling and concatenation - building universal dataset ##

def getlabelintercept(Labels):
    u,c=np.unique(np.concatenate(Labels),return_counts=True)
    Lint=u[np.where(c==len(Labels))[0]] 
    return Lint   

def unionlabeltraj(traj_obj,label_union):
    tst=np.zeros([traj_obj.traj.shape[0],label_union.shape[0]]).T
    for i,k in enumerate(np.where(np.in1d(label_union,traj_obj.labels)==True)[0]):
        tst[k]=traj_obj.traj[:,i]
    return Traj(label_union,tst.T.astype(bool))

def getuniontrajs(trajs):
    U=np.sort(np.unique(np.concatenate([m.labels for m in trajs])))
    return [unionlabeltraj(m,U) for m in trajs]

def sampledata(data,samplesize,ts=None,before=2500,after=2500):  
    if ts==None:
        sampindex=np.random.choice(np.arange(data.shape[0]),size=samplesize,replace=True)
    if ts is not None:   
        sampindex=np.random.choice(np.arange(ts-before,ts+after),size=samplesize,replace=True)
    return data[sampindex,:]

def getuniversaldataset(trajs,samplesize=200,concat=False,union=False):
    Lint=getlabelintercept([trajs[i].labels for i in range(len(trajs))])
    if concat is False:
        T=Traj(Lint,np.concatenate([sampledata(trajs[i].traj[:,np.in1d(trajs[i].labels,Lint)],samplesize=samplesize) for i in range(len(trajs))]))
        T.remove_singles()
        return T
    T=Traj(Lint,np.concatenate([np.column_stack((trajs[i].traj[:,np.in1d(trajs[i].labels,Lint)],np.ones(len(trajs[i].traj))*i)) for i in range(len(trajs))]))
    T.remove_singles()
    return T



## Time-resolved Bayesian network Analysis ##



def getedgefromdot(dotfile,moralize=False):
    G=nx.DiGraph(read_dot(dotfile))
    nodes=np.sort(np.array(G.nodes()))
    if moralize==True:
        mG=moral_graph(G)
        edges=np.array([e[0]+'->'+e[1] for e in np.array(list(mG.edges()))])
        return nodes,edges
    edges=np.array([e[0]+'->'+e[1] for e in  np.array(G.edges())])
    return nodes,edges

# create library(dictionary) for labels(nodes)
def getlabdict(nodes):
    labdict={}
    for index,element in enumerate(nodes):
        labdict[element]=index
    return labdict

def edgeenumerate(edgenames,labdict):
    edgesplit=np.array([i.split('->') for i in edgenames])
    return np.array([[labdict[i],labdict[j]] for i,j in edgesplit])

def getscanWindows(datamax,window,shift):
    x=np.arange(0,datamax-window,shift)
    y=x+window
    return np.column_stack([x,y])

def miTimeScan(data,edgenums,w):
    return [mi_p([data[w[0]:w[1],i],data[w[0]:w[1],j]]) for i,j in edgenums]

def miSpaceScan(data,edgenums,kernal):
    return [mi_p((data[kernal,i],data[kernal,j])) for i,j in edgenums]

class Scan():
    def __init__(self,data,data_labels,dotfile,moral=False,windowlist=None,kernals=None):
        #Initialize class with input trajectory data and data_labels
        self.inputtraj=data.astype(int)
        self.inputlabels=data_labels
        # get edges and nodes from dotfile and enumerate edges
        self.nodes,self.edges=getedgefromdot(dotfile,moralize=moral)
        self.data=data[:,np.in1d(self.inputlabels,self.nodes)].astype(int)
        self.datamax=self.data.shape[0]
        self.nodedict=getlabdict(self.nodes)
        #print(self.data.shape,self.nodes.shape,self.edges.shape)
        self.edgenums=edgeenumerate(self.edges,self.nodedict)

        # Prepare windows list 
        self.windowlist=windowlist
       
       # Prepare Iterator if Spatial
        self.kernals=kernals        
 
        # Create networkx graph
        self.G=nx.drawing.nx_agraph.read_dot(dotfile)
        self.G_edgevals=np.array([list(self.G.edges.data())[i][2]['label'] for i in range(len(list(self.G.edges)))]).astype(float)

    def timescores(self,window):
        return miTimeScan(self.data,self.edgenums,window)

    def stscores(self,i):
        return miSpaceScan(self.data,self.edgenums,i)

    def settm3tm6(self,tm3tm6_file=None,tm3tm6_data=None):
        if tm3tm6_data is not None:
            self.tm3tm6=tm3tm6_data
            return
        self.tm3tm6=np.load(tm3tm6_file)    

    def settracks(self,tracks):
        self.tracks=tracks

    def computewd(self):
        src=[np.where(self.edgenums==i)[0] for i in range(self.nodes.shape[0])]
        self.wdegree=np.array([[np.sum(self.tracks[src[i],t]) for t in range(self.tracks.shape[1])] for i in range(self.nodes.shape[0])])

    def maxwd(self):
        return np.array([np.amax(i) for i in self.wdegree])

    def wdsort(self):
        self.wdsort=np.flip(np.argsort(self.maxwd()))

def scanandsave(scan,nprocs,scoresdir='./masterscan/'):
    scores=[]
    if scan.windowlist is not None:
        for i,window in enumerate(scan.windowlist):
            W=getscanWindows(scan.datamax,window,shift=1)
            out=np.array(runParallel(scan.timescores,W,nprocs))
            scores.append(np.array(out).T)
        return scores
    for i,kernal in enumerate(scan.kernals):
        out=np.array(runParallel(scan.stscores,kernal,nprocs))
        scores.append(out)
    return np.array(scores)

def gettrajdbns(trajs,bn_dot='./bn.dot',windowlist=[50,100,200,400],nprocs=4,save_S=False):
    D=[Scan(t.traj,t.labels,bn_dot,windowlist=windowlist) for t in trajs]    
    print(f'Scanning trajectories using Universal BN with:\n{len(D[0].nodes)} nodes & {len(D[0].edges)} edges')
    S=[scanandsave(d,nprocs=nprocs) for d in D]
    if save_S:
        picklewrite('S.pkl',S)
        return
    print('Performing Smoothing')
    Dots=[Scandot(s,windowlist,D[i].data.shape[0]) for i,s in enumerate(S)]
    Outs=[[d.allwinalledge(j) for j in range(len(windowlist))] for d in Dots]
    Tracs=[getalltracks(Dots[i].converttoheatmap(Outs[i])) for i in range(len(Dots))]
    [d.settracks(Tracs[i]) for i,d in enumerate(D)]
    [d.computewd() for d in D]
    [d.wdsort() for d in D]
    print('Complete!')
    return D

## Trajectory scan output smoothing ##

def getT(w,timearr):
    return((timearr[:len(timearr)-w][:,None]<=timearr) & (timearr[:len(timearr)-w][:,None]+w>timearr)).astype(int)

def loopoverwindow(time,windows):
    tarr=np.arange(time)
    return [getT(w,tarr) for w in windows]

class Scandot():
    def __init__(self,tracks,windows,time):
        self.time=time
        self.windows=windows
        self.T=loopoverwindow(time,windows)
        if tracks[0].shape[1] != self.T[0].shape[0]:
            self.tracks=[i.T for i in tracks]
        self.tracks=tracks
        #print(self.tracks[0].shape,self.T[0].shape)
    def allwinalledge(self,window):
        u=self.T[window].sum(0)
        u[-1]=1
        return np.dot(self.tracks[window],self.T[window])/u

    def converttoheatmap(self,dotout):
        return np.array([[dotout[i][j] for i in np.arange(len(self.windows))] for j in range(dotout[0].shape[0])])

def getmaxfromheat(heatmap):
    return np.array([np.argmax(np.amax(i,axis=1)) for i in heatmap])

def getalltracks(heatmap,peakth=0.01):
    maxargs=getmaxfromheat(heatmap)
    alltracks=np.array([heatmap[e][maxargs[e]] for e in range(len(maxargs))])
    return alltracks

def scanandupdate(dbn):
    scores=scanandsave(dbn,nproc=4)
    scan=Scandot(scores,dbn.windowlist,dbn.data.shape[0])    
