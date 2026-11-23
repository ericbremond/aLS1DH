
import numpy as np
from scipy.linalg import cho_solve

from qml import kernels
from cMBDF import generate_mbdf, get_convolutions
from KernelRidge import GridSearchCV_local

import os, sys

def rcsv(fname):

    energies = []; data = []
    for line in open(fname).readlines()[1:]:
        words = line.split('"')
        energies.append(float(words[5])-float(words[7]))
        data.append([words[1], float(words[7])])

    return energies, data

def read_orca(fname):

    at2q = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
        'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
        'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
        'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86,
        'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
        'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103,
        'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
        'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }

    charges = []; xyzs = []
    c = 0; m = 1; ne = 0
    with open(fname) as pfname:
        for line in pfname:
            if line.find("*xyz") >= 0:
                words = line.split()
                c = int(words[1])
                m = int(words[2])

                line = next(pfname)
                while "end" not in line:
                    words = line.split()
                    charges.append(at2q[words[0]])
                    xyzs.append([float(words[1]), float(words[2]), float(words[3])])

                    ne += at2q[words[0]]

                    line = next(pfname)

    return ne-c, np.array(charges), np.array(xyzs)

def padding(a, pad = None):

    if pad == None:
         pad = max(len(ai) for ai in a)

    return np.array([list(ai) + [0] * (pad - len(ai)) for ai in a])

def run_KRR(xtest, qtest, ntest, npzname, debug = False):

    with np.load(npzname) as data:
        alpha = data['alpha']
        xtrain = data['xtrain']
        pad_qtrain = data['qtrain']
        sigma = data['sigma']
        L = data['L']

    pad_qtest = padding(qtest)

    k = kernels.get_local_kernels(xtrain, xtest, pad_qtrain, pad_qtest, [sigma])[0]

    y_pred_scaled = np.dot(k, alpha) / ntest

    return y_pred_scaled

def worca(fname, ax, crds):

    f = open(fname, "w")

    f.write("%pal\n")
    f.write("  nprocs       16\n")
    f.write("  end\n")
    f.write("\n")
    f.write("%MaxCore     10000\n")
    f.write("\n")
    f.write("! RKS def2-QZVPP RIJCOSX AutoAux TightSCF DEFGRID3\n")
    f.write("\n")
    f.write("%method\n")
    f.write("  Exchange X_PBE\n")
    f.write("  Correlation C_PBE\n")
    f.write("  ScalHFX = %.4f\n"%(ax))
    f.write("  ScalDFX = %.4f\n"%(1-ax))
    f.write("  ScalGGAC = %.4f\n"%(1-ax**3))
    f.write("  ScalLDAC = %.4f\n"%(1-ax**3))
    f.write("  ScalMP2C = %.4f\n"%(ax**3))
    f.write("  end\n")
    f.write("\n")
    f.write(crds)
    f.write("end\n")
    f.write("\n")
    f.close()

    return 0

def main():

    ycenter = 0.5

    # read orca input files
    q_run = []; n_run = []; r_run = []
    for fname in sys.argv[1:]:
        ne, qs, xyzs = read_orca(fname)
        n_run.append(ne)
        q_run.append(qs)
        r_run.append(xyzs)

    # generate cMBDF descriptor
    convs = get_convolutions(rcut = 6.0)
    x_run = generate_mbdf(q_run, r_run, convs, n_atm=2.0)

    # evaluate (ax, ac) pairs
    dax = run_KRR(x_run, q_run, n_run, "KRR_qm5_qm9.npz", debug = True)

    # write aLS1DH orca input
    for i, fname in enumerate(sys.argv[1:]):
        crds = ""
        with open(fname) as pfname:
            for line in pfname:
                if line.find("xyz") >= 0:
                    while "end" not in line:
                        crds += line
                        line = next(pfname)
        
        worca("aLS1DH_%s"%(fname), dax[i]+ycenter, crds)

    return 0

if __name__ == "__main__":

    main()

(qml) [eric@batwoman aLS1DH]$ /cho_so
-bash: /cho_so: No such file or directory
(qml) [eric@batwoman aLS1DH]$ vi aLS1DH.py 
(qml) [eric@batwoman aLS1DH]$ vi aLS1DH.py 
(qml) [eric@batwoman aLS1DH]$ cat aLS1DH.py 

import numpy as np

from qml import kernels
from cMBDF import generate_mbdf, get_convolutions
from KernelRidge import GridSearchCV_local

import os, sys

def rcsv(fname):

    energies = []; data = []
    for line in open(fname).readlines()[1:]:
        words = line.split('"')
        energies.append(float(words[5])-float(words[7]))
        data.append([words[1], float(words[7])])

    return energies, data

def read_orca(fname):

    at2q = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
        'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
        'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
        'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86,
        'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
        'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103,
        'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
        'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }

    charges = []; xyzs = []
    c = 0; m = 1; ne = 0
    with open(fname) as pfname:
        for line in pfname:
            if line.find("*xyz") >= 0:
                words = line.split()
                c = int(words[1])
                m = int(words[2])

                line = next(pfname)
                while "end" not in line:
                    words = line.split()
                    charges.append(at2q[words[0]])
                    xyzs.append([float(words[1]), float(words[2]), float(words[3])])

                    ne += at2q[words[0]]

                    line = next(pfname)

    return ne-c, np.array(charges), np.array(xyzs)

def padding(a, pad = None):

    if pad == None:
         pad = max(len(ai) for ai in a)

    return np.array([list(ai) + [0] * (pad - len(ai)) for ai in a])

def run_KRR(xtest, qtest, ntest, npzname, debug = False):

    with np.load(npzname) as data:
        alpha = data['alpha']
        xtrain = data['xtrain']
        pad_qtrain = data['qtrain']
        sigma = data['sigma']
        L = data['L']

    pad_qtest = padding(qtest)

    k = kernels.get_local_kernels(xtrain, xtest, pad_qtrain, pad_qtest, [sigma])[0]

    y_pred_scaled = np.dot(k, alpha) / ntest

    return y_pred_scaled

def worca(fname, ax, crds):

    f = open(fname, "w")

    f.write("%pal\n")
    f.write("  nprocs       16\n")
    f.write("  end\n")
    f.write("\n")
    f.write("%MaxCore     10000\n")
    f.write("\n")
    f.write("! RKS def2-QZVPP RIJCOSX AutoAux TightSCF DEFGRID3\n")
    f.write("\n")
    f.write("%method\n")
    f.write("  Exchange X_PBE\n")
    f.write("  Correlation C_PBE\n")
    f.write("  ScalHFX = %.4f\n"%(ax))
    f.write("  ScalDFX = %.4f\n"%(1-ax))
    f.write("  ScalGGAC = %.4f\n"%(1-ax**3))
    f.write("  ScalLDAC = %.4f\n"%(1-ax**3))
    f.write("  ScalMP2C = %.4f\n"%(ax**3))
    f.write("  end\n")
    f.write("\n")
    f.write(crds)
    f.write("end\n")
    f.write("\n")
    f.close()

    return 0

def main():

    ycenter = 0.5

    # read orca input files
    q_run = []; n_run = []; r_run = []
    for fname in sys.argv[1:]:
        ne, qs, xyzs = read_orca(fname)
        n_run.append(ne)
        q_run.append(qs)
        r_run.append(xyzs)

    # generate cMBDF descriptor
    convs = get_convolutions(rcut = 6.0)
    x_run = generate_mbdf(q_run, r_run, convs, n_atm=2.0)

    # evaluate (ax, ac) pairs
    dax = run_KRR(x_run, q_run, n_run, "KRR_qm5_qm9.npz", debug = True)

    # write aLS1DH orca input
    for i, fname in enumerate(sys.argv[1:]):
        crds = ""
        with open(fname) as pfname:
            for line in pfname:
                if line.find("xyz") >= 0:
                    while "end" not in line:
                        crds += line
                        line = next(pfname)
        
        worca("%s_aLS1DH.inp"%(fname[:-4]), dax[i]+ycenter, crds)

    return 0

if __name__ == "__main__":

    main()

