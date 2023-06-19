# Creates repeated structural beta estimates which are required to generate confidence intervales for the 
# impulse responses of the interaction Panel VAR. The program generates bootstrapped structural beta coefficients 
# and errors by simulating repeatedly the dataset and reestimating the structural coefficients on the 
# bootstrapped samples. The program requires various inputs also specifieic for the interactPVAR.m program 
# to estimate the point estimates of the original data. The output can be used to contruct confidence intervals 
# for the point estimates of interactPVAR.
#
# Inputs include: 
# - parametric: is a 0 - 1 variable where 1 stands for parametric errors and 
#               0 for non paramtric error drawings
# - positioner:  Indicates where the errors for a new country start and begin
# - Idata:      The interaction terms, matrix of dimension (I*times - by - # of interactions)
# - ydata:      The y-variable matrix used in the estimation of interactPVAR
# - errors:     The structural errors (output from interactPVAR.m)
# - beta:       The structural beta estimates as produced by interactPVAR.m 
# - number:     Is the number of draws for restimation (usually 100-500). 
#               The fewer the faster, though the less accurate.
# - I:          Number of countries (=1 implies one country -> no panel)
# - lag:        Number of lags used in the estimation
# - restr:      The matrix of restrictions specified for the estimation
#               See interactPVAR.m
# - hominterc:  for the first hominterc equations interaction have no
#               effect on levels (no interaction with constant) 
# - demean:     = 0 Do not demean data = 1 demean data (fixed effects)
#
# 
# Outputs incldue:
# - BETAMAT:    The matrix of structural beta estimates of dimension 
#               [size(beta,1)), np.shape(beta)[1],number]
# - ERRORMAT:   The matrix of structural errors of dimension 
#               [size(errors,1)), size(errrors,2),number]
# - STERRMAT:   The matrix of structural beta estimates of dimension 
#               [size(beta,1)), np.shape(beta)[1],number]
#-----------------------------------------------------------------------------
# Calls on the functions: interactPVAR.m , gapfill.m
#=============================================================================
# See also interactPVAR, irfpreppvar, gapfill  
#
# Written by Pascal Towbin and Sebastian Weber - 20.02.2009
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np
import random
from gapfill import gapfill
from paneldetrend import paneldetr
from interactPVAR import interactPVAR

def IPVARboot(parametric,positioner,Idata,ydata,errors,beta,number,I,lag,restr,hominterc,demean):
    #
    # DETERMING ESTIMATION APPROACH
    #-------------------------------------------------------------------------------------------------------------------
    method1 = 'nearest'
    method2 = 'nearest'
    xdata = ydata
    
    time = np.shape(ydata)[0]/I
    nint = np.shape(Idata)[1]
    
    if not time == np.floor(time):
        print('Error: The time dimension is not identical across individuals!!')
    time = int(time)
    
    #allocate space
    BETAMAT  = np.zeros((np.shape(beta)[0],np.shape(beta)[1],number)) 
    ERRORMAT = np.zeros((np.shape(errors)[0],np.shape(beta)[1],number)) 
    STERRMAT = np.zeros((np.shape(beta)[0],np.shape(beta)[1],number)) 
    
    ROWIND_Y = [None]*np.shape(beta)[1]; ROWIND_X = [None]*np.shape(beta)[1]
    # Find NaNs
    #--------------------------------------------------------------------------
    for i in range(np.shape(beta)[1]):   
       ROWIND_Y[i] = np.argwhere(np.isnan(ydata[:,i]))[:,0]
       ROWIND_X[i] = np.argwhere(np.isnan(xdata[:,i]))[:,0]
    
    
    # CONSTRUCTING ARTIFICIAL DATA
    #--------------------------------------------------------------------------
    Ty, nvar = np.shape(ydata)
    Tx, Kx   = np.shape(xdata)
    Ti, Ki   = np.shape(Idata)
    
    Xn = [None]*lag; Vn = [None]*lag
    Xhelp = [None]*I;   Vhelp = [None]*I
    Xn    = [None]*lag; Vn    = [None]*lag
    XX    = [None]*lag; VV    = [None]*lag
    for i in range(lag):
        for j in range(I):
            X = None; V = None
            for nx in range(Kx):
                Xh = None; Vh = None
                for k in range(Ki):
                    start = lag+j*time
                    ende  = (1+j)*time
                    if Xh is None:
                        Xh = xdata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,k] 
                    else:
                        Xh = np.column_stack([Xh, xdata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,k]])  
                    
                Xh = np.column_stack([xdata[i+j*time:(1+j)*time-lag+i,nx], Xh])
                if Vh is None:
                    Vh = xdata[i+j*time:(1+j)*time-lag+i,nx]
                else:
                    Vh = np.column_stack([xdata[i+j*time:(1+j)*time-lag+i,nx], Vh])
                
                if X is None:
                    X = Xh
                else:
                    X = np.column_stack([X, Xh])
                if V is None:
                    V = Vh
                else:
                    V = np.column_stack([V, Vh])
            
            Xhelp[j] = X
            Vhelp[j] = V
            if Xn[i] is None:
                Xn[i] = Xhelp[j]
            else:
                Xn[i] = np.vstack([Xn[i], Xhelp[j]])
            if Vn[i] is None:
                Vn[i] = Vhelp[j]
            else:
                Vn[i] = np.vstack([Vn[i], Vhelp[j]])
        
        XX[i] = Xn[i]
        VV[i] = Vn[i]

    # Create Regressors (NOTE: THE ORDER IS REVERSED)
    X = XX[0];  V = VV[0]
    for i in range(1,lag):
        X = np.hstack([XX[i], X])   
        V = np.hstack([VV[i], V])   
    
    # Demean Regressors
    if demean == 1: # if estimations is fixed effect
        for columni in range(np.shape(X)[1]):
            X[:,columni] = paneldetr(X[:,columni],I,0)

        for columni in range(np.shape(V)[1]):
            V[:,columni] = paneldetr(V[:,columni],I,0)

    #else:  # if estimations is not fixed effect

    # CREATING BETAS
    Bbeta = [None]*nvar
    #--------------------------------------------------------------------------
    for eqnum in range(nvar):
        start = (nvar-eqnum-1)*(nint+1)
        ende = len(beta)-(1+nint)
        Bbeta[eqnum] = beta[start:ende,eqnum]
  
    art_data_all = np.zeros((I*time,nvar))
    # START BOOTSTRAP
    #-----------------------------------------------------------------------
    for nobs in range(number):
       # DRAW NEW ERRORS from OLD ONES if parametric =0 else from std normal
       #--------------------------------------------------------------------
       
       if parametric == 0:
           c_errors = errors
       else:           
           c_errors = np.random.normal(0,np.kron(np.ones((len(errors),1)),np.std(errors,0)))
      
       for i in range(I):
            # Drawing ERRORS
            if i == 0 and positioner[i,1] == 0:
                new_data = np.empty((time,nvar))   
            elif i > 0 and positioner[i-1,1] == positioner[i,1]:
                new_data = np.empty((time,nvar)) 
            else:
                new_data = np.empty((time,nvar))   
                c_new_error = [None]*time; int_terms = np.empty((time,nint))
                # Interpolation of dummies if not full
                for k in range(nint):
                    tmp = np.sum(~np.isnan(Idata[i*time:(1+i)*time,k]))
                    if tmp > 1:
                        intp = gapfill(Idata[i*time:(1+i)*time,k],method1,method2)
                        int_terms[:,k] = intp
                    else:
                        Ivalue = Idata[i*time:(1+i)*time,k]
                        ind = np.argwhere(~np.isnan(Ivalue))[:,0]
                        if len(ind) > 0:
                            int_terms[:,k] = Ivalue[ind]*np.ones(np.shape(Idata[i*time:(1+i)*time,k]))
                        else:
                            int_terms[:,k] = []
                            
                # DRAWING ERRORS
                population = range(len(c_errors))
                c_indicator = random.choices(population,k=time)                                             
                for j in range(len(c_indicator)): 
                    c_new_error[j] = c_errors[c_indicator[j]]
                           
                # DRAWING INITIAL OBSERVATION
                counter = 0
                while np.any(np.isnan(X[counter+i*(time-lag)])):
                    counter += 1
       
                art_data = np.transpose(X[counter+i*(time-lag)])
                v_data = np.transpose(V[counter+i*(time-lag)])
        
                # SIMULATION      
                for j in range(len(c_new_error)):   
                       art_dataH  =  art_data.copy()
                       tmp = np.array([1] + int_terms[j].tolist()) 
                       current_line = []
                       for eqnum in range(nvar):             
                            art_val  =  np.transpose(art_dataH) @ Bbeta[eqnum] + c_new_error[j][eqnum]
                            new_data[j,eqnum] = np.transpose(art_val)       # creating new data as in standard format
                            # Update old data and creation of updated interactions
                            current_eq = tmp * art_val
                            current_line += current_eq.tolist()
                            art_dataH = np.concatenate([current_eq, art_dataH])
                           
                       t1 = np.kron(new_data[j,:],np.transpose(tmp))
                       t1 = np.transpose(t1)
                       t2 = np.kron(v_data[:nvar*(lag-1)],np.transpose(tmp))
                       t2 = np.transpose(t2)
                       art_data = np.concatenate([t1,t2])
                      
                       #art_data = [np.kron(new_data[j,:],np.transpose([1, int_terms[j,:]) @ np.transpose([kron(v_data[1:nvar*(lag-1),0])]) ,np.transpose([1, int_terms[j,:]])]
                       v_data = np.concatenate([np.transpose(new_data[j,:]), v_data[:nvar*(lag-1)]])

            art_data_all[i*time:(1+i)*time] = new_data

       art_data = art_data_all
        
       for i in range(nvar):
           no_nan_y = len(ROWIND_Y[i])
           ydata[:,i] = art_data[:,i]
           for k in range(no_nan_y):  
               ydata[ROWIND_Y[i][k],i] = np.nan 
        
           no_nan_x = len(ROWIND_X[i])
           xdata[:,i] = art_data[:,i]
           for k in range(no_nan_x):  
               xdata[ROWIND_X[i][k],i] = np.nan 
      
       # REPEAT VAR ESTIMATION
       #-----------------------------------------------------------------------------
       betaX,sterrX,errorsX,positioner,Fvalue = interactPVAR(ydata,I,lag,Idata,hominterc,restr,demean)
        
       # STORE RESULTS
       #----------------------------------------------------------------------------
       BETAMAT[:,:,nobs] = betaX 
       ERRORMAT[:,:,nobs] = errorsX 
       STERRMAT[:,:,nobs] = sterrX 
        
       sterrX = []; betaX = []; errorsX = []
       ### end of bootstrapping
    
    return BETAMAT, ERRORMAT, STERRMAT