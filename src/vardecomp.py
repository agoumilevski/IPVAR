# Based on the Matlab code of Pascal Towbin and Sebastian Weber
# Translated to Python by A. Goumilevski - 05.01.2023

import numpy as np

def vardecomp(stepsahead,MA):
    """ Produces steps ahead variance decomposition
        MA is the data as given by PVAR.
    """
    
    ni,nj,nk = np.shape(MA)
    nvar = nj
    
    # IRFs by period first column is the first equations shock row is response
    irf = np.zeros((nvar,nvar,stepsahead))
    
    
    for j in range(nj):
        for k in range(nk):
            for i in range(ni):
                irf[j,k,i] = MA[i,j,k]
    
    squareirf = np.zeros((nvar,nvar,stepsahead))
    for j in range(stepsahead):
        for k in range(nvar):
                for i in range(nvar):
                    squareirf[i,k,j] =irf[i,k,j]**2
    
    
    # summing acroos steps ahead:
    sumsquareirf = np.zeros((nvar,nvar,stepsahead))
    for j in range(stepsahead):
       for i in range(nvar):
              for k in range(nvar):
                  sumsquareirf[i,k,j] =sum(squareirf[i,k,:j])
    
    # summing accros shocks
    Ssumsquareirf = np.zeros((stepsahead,nvar))
    for j in range(stepsahead):
       for i in range(nvar):
           Ssumsquareirf[j,i] = np.sum(sumsquareirf[i,:,j])
    
    # VAR DECOMP
    decomp = np.zeros((stepsahead,nvar,nvar))
    for j in range(stepsahead):
       for i in range(nvar):
              for k in range(nvar):
                  decomp[j,k,i] = sumsquareirf[i,k,j]/max(1.e-20,Ssumsquareirf[j,i])
    
    
    return decomp


