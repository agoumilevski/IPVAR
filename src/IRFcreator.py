# Creates Impulse Response functions from reduced form betas 
# as retured by PVAR.m or irfpreppvar.m It is also a sub-routine 
# of IRF_IPVAR.m
#
# Inputs include:
# - beta:    beta coefficient as returned by PVAR.m or irfpreppvar.m
#            where the form is given by a matrix of dimension
#            (# of var * # of lags) - by - nvar
# - Pkomega: The Covariance matrix as given by PVAR.m or irfpreppvar.m
# - lag:     # of lags used i nthe estimation
# - period:  Wished lentgh of the period for which the IRF is computed
#
# Output:
# - IRF:     The imulse responses, a matrix of dimension: 
#            (# of periods,# of responses, # of shcoks)
# ==========================================================================
# See also IRF_IPVAR.m, irfpreppvar, PVAR, IRFcreator2
#
# Written by Pascal Towbin and Sebastian Weber - 20.02.2009
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np

def IRFcreator(beta,Pkomega,lag,period):
    
    nvar = np.shape(beta)[1]
    
    res = [None]*nvar; res2 = [None]*nvar
    for k in range(nvar):
        res[k] = beta[:,k]

    # Ordering beta by variable (not by lag), i.e. like in EViews OLS output 
    for k in range(nvar):
        tmp = np.zeros(lag*nvar)
        for j in range(lag): 
            for i in range(nvar): 
                tmp[j+lag*i] = res[k][j*nvar+i]
        res2[k] = tmp.tolist()
        
    for k in range(nvar):
        res2[k].extend(res[k][nvar*lag:]) 

    # Bringing Betas in shape (nvar*nlag  nvar*nlag nvar*nlag as many times as
    # there are variables)
    betall = np.array(res2)
   
    # Creating blocks by lag order
    betablock = np.zeros((nvar,nvar,lag))
    for i in range(nvar):
        for j in range(lag):
            betablock[:,i,j] = betall[:,i*lag+j]
    
    # Deriving MA representation
    h2rho  =  np.zeros((nvar,nvar,period))
    #MA(:,:,:) = zeros(period,nvar,nvar)
    #MA(1,1:nvar,1:nvar) = Pkomega 21.1.2009
    rho  =  np.zeros((nvar,nvar,period))
    rho[:,:,0] = np.eye(nvar)
    hrho  =  rho.copy()
    
    for i in range(period-1):
        j = 0
        while j <= i:   
            if 1+i-j > lag:
                hrho[:,:,1+j] = 0
            else:
                hrho[:,:,1+j] = rho[:,:,j] @ betablock[:,:,i-j]
            h2rho[:,:,j] = hrho[:,:,1+j]
            j += 1
        rho[:,:,1+i] = np.sum(h2rho,2)
        
    # Storing IRFs
    IRF = np.zeros((period,nvar,nvar))
    for i in range(period):        
        IRF[i] = rho[:,:,i] @ Pkomega # in std dev.

    return IRF