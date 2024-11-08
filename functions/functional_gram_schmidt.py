from copy import deepcopy

def functional_gram_schmidt(finp):
    '''
    receives a list of functions
    '''
    fout = []
    d = len(finp)
    for i in range(d):
        f = deepcopy(finp[i])
        for j in range(i):
            f = f + finp[i].project(fout[j]).multiply(-1)
        f.normalize()
        fout.append(f)
    return fout
