#=================================================================== 
# Evaluates the beta ceofficients that are the output of the 
# interactPVAR.m command at the prespecified values for the inter-
# action terms.
# Required inputs are
# - beta:   output of the interactpvar (m-by-n)
# - values: prespecified values for the interaction terms. Matrix of 
#           dimension: (# of evaluations - by- # of interactions)  
# - lag:    # of lags in the estimation 
#===================================================================
#
# See also irfpreppvar, interactPVAR  
#
# Written by Pascal Towbin and Sebastian Weber - 20.02.2009
# Translated to Python by A. Goumilevski - 05.01.2023

import numpy as np

def evalpvar(nint,beta,values,lag):
    
    neq = np.shape(beta)[1]		 # number of equations
    values = np.array([1] + values)
    #print(values)
    
    # Evaluates the betas at the wished values 
    betaeval = [None]*(1+lag)
    betaeval2 = [None]*(1+lag)
    constanteval = np.empty(neq)
    
    betaeval[0] = beta[:(1+nint)*(neq-1)].copy()
    
    for lg in range(lag):
        n1 = (1+nint)*(neq-1)+((1+nint)*neq)*lg
        n2 = (1+nint)*(neq-1)+((1+nint)*neq)*(1+lg)
        betaeval[1+lg] = beta[n1:n2].copy()

    b = []
    for k in range(neq):
        be = []
        for i in range(neq-1):
            n1 = (1+nint)*i
            n2 = (1+nint)*(1+i)
            x = betaeval[0][n1:n2,k] @ values
            be.append(x)
        b.append(be)
    betaeval2[0] = np.transpose(np.array(b))
    
    for lg in range(lag):
        b = np.empty((neq,neq))
        for i in range(neq):
            for k in range(neq):
                n1 = (1+nint)*i
                n2 = (1+nint)*(1+i)
                b[i,k] = betaeval[lg+1][n1:n2,k] @ values
        betaeval2[1+lg] = b
    
    evalbeta = betaeval2[0]
    for lg in range(1,lag+1):
        evalbeta = np.vstack((evalbeta,betaeval2[lg]))
    
    for i in range(neq):
        constanteval[i] = beta[-nint-1:,i] @ values

    evalbeta = np.vstack((evalbeta,constanteval))
    
    return evalbeta
