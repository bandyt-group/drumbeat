import numpy as np

def getlastline(file):
    with open(f'{file}', 'rb') as f:  # Open in binary mode for precise byte-level control
        f.seek(-2, 2)  # Move to the second-to-last byte in the file
        while f.read(1) != b'\n':  # Move backward until a newline is found
            f.seek(-2, 1)          # Move 2 bytes back each iteration
        last_line = f.readline().decode()  # Read and decode the last line

    return last_line

def getC(file,frames):
    f=open(file)
    f.readline().strip().split('\t')
    f.readline().strip().split('\t')
    C=[]
    for i in range(frames):
        if i==0:
            line=f.readline().strip().split('\t')
            time=int(line[0])
        contact=line[2].split(':')[1]+line[2].split(':')[2]+'_'+line[3].split(':')[1
    ]+line[3].split(':')[2]
        contacts=[]
        contacts.append(contact)
        while time==i:
            line=f.readline().strip().split('\t')
            if len(line)==1:    #End of File
                break 
            time=int(line[0])
            if time==i:
                contacts.append(line[2].split(':')[1]+line[2].split(':')[2]+'_'+line
    [3].split(':')[1]+line[3].split(':')[2])
        C.append(set(contacts))
    return C


def gettrajfromtsv(file,tmax=None):
    if tmax is None:
        tmax=int(getlastline(file).split('\t')[0])+1
    C=getC(file,tmax)
    labels=np.unique((np.concatenate([np.array(list(c)) for c in C])))
    traj=np.array([np.in1d(labels,np.array(list(c))) for c in C])
    return labels,traj
