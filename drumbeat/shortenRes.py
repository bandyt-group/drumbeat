import numpy as np

class Residue():
    def __init__(self):
        self.d={'CYS': 'C', 'CYX': 'C', 'ASP': 'D','ASH': 'D' ,'SER': 'S', 
                'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 
                'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H','HSD': 'H',
                'HID': 'H' ,'HIE': 'H', 'HIP': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                'ALA': 'A', 'VAL':'V', 'GLU': 'E','GLH': 'E' , 'TYR': 'Y', 'MET': 'M',
		'ZD7': 'Z', 'NMA': 'NM' }

    def shRes(self,x):
        #print(self.d.keys())
        if len(x) % 3 != 0: 
            raise ValueError('Input length should be a multiple of three')

        y = ''
        for i in range(len(x) // 3):
            y += self.d[x[3 * i : 3 * i + 3]]
        return y

