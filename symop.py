from os.path import dirname,realpath
import numpy as np
import re

class symops(dict):
    def __init__(self, libFN=None):
        if libFN is None:
            libFN = dirname(realpath(__file__)) + "/symop.lib"
        self._parse(libFN)
    def _parse(self, libFN):
        with open(libFN, 'r') as f:
            for match in re.findall(r"(?<=\n)[0-9].*?(?=\n[0-9])", '\n'+f.read(), re.DOTALL):
                k = int(match.split()[0])
                self[k] = symop(match)

class symop(dict):
    def __init__(self, text):
        self.number = int(text.split()[0])
        self.name = re.findall(r"(?<=').*?(?=')", text)[0]
        self._parse(text)
    def _parse(self, text):
        for line in text.split('\n')[1:]:
            self[line] = op(line)

class op():
    def __init__(self, text):
        self.rot_mat = np.zeros((3,3))
        ax  = { 
            'X':  np.array([1., 0., 0.]), 
            'Y':  np.array([0., 1., 0.]), 
            'Z':  np.array([0., 0., 1.]),
           }

        for i,t in enumerate(text.split(',')):
            for k,v in ax.items():
                if '-' + k in t:
                    self.rot_mat[:,i] -= v
                elif k in t:
                    self.rot_mat[:,i] += v

        self.trans = np.zeros(3)
        div = lambda x: float(x[0])/float(x[1])
        x,y,z = text.split(',')
        self.trans[0] = 0. if '/' not in x else div(re.sub(r"[^\/0-9]", "", x).split('/'))
        self.trans[1] = 0. if '/' not in y else div(re.sub(r"[^\/0-9]", "", y).split('/'))
        self.trans[2] = 0. if '/' not in z else div(re.sub(r"[^\/0-9]", "", z).split('/'))
    def __call__(self, vector):
        return np.matmul(self.rot_mat, vector)

    def translate(self, vector):
        """
        There is a decent chance this is garbage. Not necessary now, but TODO: fix this
        """
        return vector + self.trans*vector

class spacegroupnums(dict):
    def __init__(self, libFN=None):
        if libFN is None:
            libFN = dirname(realpath(__file__)) + "/symop.lib"
        self._parse(libFN)
    def _parse(self, libFN):
        with open(libFN, 'r') as f:
            for line in f:
                if line[0] != ' ':
                    k = line.split("'")[1]
                    v = int(line.split()[0])
                    self[k] = v

symops = symops()
spacegroupnums = spacegroupnums()
