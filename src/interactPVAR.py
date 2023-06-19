#===================================================================================== 
# Program produces structural beta corefficents of a Panel VAR which allows for
# interaction terms. It allows interaction terms but imposes a Choleski structure. 
# Hence, the data needs to be entered according to the ordering.
#
# Inputs are:
# - ydata:   The endogenous variables
# - I:       Number of cross-sectional units (e.g. number of countries)
# - lag:     Number of lags to be used in the estimation
# - Idata:   Interaction Terms used inthe estimation. A matrix of dimnesion
#            [I*time - by - # of interaction variables] 
# - hominterc:  for the first hominterc equations interaction have no
# effect on levels (no interaction with constant)
# - restr:   Is a matrix of size nvar*(nvar*nint+1) with zeros and ones defining for 
#            which equation which interaction and variable a restriction is supposed 
#            to be applied (ones leading to an exclusion of the interaction/or variable).
#            If there is an interaction in one period it is assumed to be relevant at all lags!
#            restr is of the following form:
#
#		         	      restr11 .    .   .   .   restn1
#			            .       .    .   .   .
#			   restr =  .       .    r   .   .
#     	      		.       .    .   .   .
#			           	rest1i  .    .   .   .   restni
#
#            Example a 2*2 VAR with two iteraction terms would possibly be
#
#			   restr = rest11   rest21  =  [0   1   1]  [1   1  1]
#			           rest12   rest22     [0   0   0]  [0   1  1]
#
#            which implies that the dynamcis of the first variable are independent of the other variable
#            and the interactions (rest11 and rest21) and only depends on its own lagged values. The second
#            (endogenous) variable's response to the the first variable can vary with the interactions (rest12) but its 
#            lagged values have always the same impact (rest22) and do not vary with the interactions.
#            Such a structure may stand for a Bi-Variate VAR of terms of trade and GDP growth, where the 
#            terms of trade are only a function of their past values and the repsonse of GDP to terms of trade
#            may vary with the exchaneg rate regime (1. Interaction) and the degree of debt (2. Interaction)
#            However, the impact of lagged values of GDP on today's GDP is independent of the exchaneg rate regime
#            and the level of debt.    
#
# - demean: = 0 Do not demean data, = 1 demean data (fixed effect) 
#
# The outputs are:
# - beta:       The structural beta coefficients which are ordered in the following way
#
#		         	      0       0    0  betan1
#			              0       0    .   . 
#			   beta  =    0       .    b   . 
#     	      		    beta      .    .   .
#			           	beta1i    .    .  betani
#
# - sterr:      The respective standard errors
# - error2:     The structural errors
# - positioner: An indicator variable which registers which observations have been used
#               for the estimation. This matters if there are missing observation and 
#               serves as input and various other programs
#====================================================================================
#
# Written by Pascal Towbin and Sebastian Weber - 20.02.2009
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np

def interactPVAR(ydata,I,lag,Idata,hominterc,restr,demean):

    
    #----------------------------------------------------------------------------------------------------
    # MANIPULATE DATA
    #---------------------------------------------------------------------------------------------------
    xdata = ydata.copy()
    Ty, nvar = np.shape(ydata)
    Tx, Kx = np.shape(xdata)
    Ti, Ki = np.shape(Idata)  # Ki - number of interaction terms
    time = np.shape(ydata)[0]/I
    if not time == np.floor(time):
        print('Error: The time dimension is not identical across individuals!!')
    time = int(time)
    
    #---------------------------------------------------------------------------------------------------------
    # CREATE INTERACTION TERMS
    # ORDERED ACCORDING TO [X1 X1*I1  X1*I2 ... X1*IKi  X2 X2*I1 ...XKx*IKi]
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------
    # CREATE REGRESSORS
    #----------------------------------------------------------------------------------------------------
    Xn = [None]*(lag+1); XX = [None]*(lag+1)

    for i in range(lag+1):
        for j in range(I):
            X = None
            for nx in range(Kx):
                Xh = None
                for ni in range(Ki):
                    start = lag+j*time
                    ende  = (1+j)*time
                    if Xh is None:
                        if i == lag+1:
                            Xh = ydata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,ni]  
                        else:
                            Xh = xdata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,ni]  
                    else:
                        if i == lag+1:
                            Xh = np.column_stack([Xh, ydata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,ni]])  
                        else:
                            Xh = np.column_stack([Xh, xdata[i+j*time:(1+j)*time-lag+i,nx]*Idata[start:ende,ni]])  
                    
                if i == lag+1:
                    Xh = np.column_stack([ydata[i+j*time:(1+j)*time-lag+i,nx], Xh])
                else:
                    Xh = np.column_stack([xdata[i+j*time:(1+j)*time-lag+i,nx], Xh])
                
                if X is None:
                    X = Xh
                else:        
                    X = np.column_stack([X, Xh])
               
            if Xn[i] is None:
                Xn[i] = X
            else:
                Xn[i] = np.row_stack([Xn[i], X])
        XX[i] = Xn[i]
    
    #  added to include interaction
    for i in range(I): 
        start = i*time
        ende  = (1+i)*time
        Idatah = Idata[lag+start:ende]
        if i == 0:
            Idata2 = Idatah
        else:
            Idata2 = np.vstack([Idata2,Idatah])
    
    # Create intercept
    X = np.ones(len(XX[0]))
    
    # Create intercept interactions
    Xinter = None
    for ni in range(Ki):
        if Xinter is None:
            Xinter = X*Idata2[:,ni]
        else:
           Xinter = np.column_stack([Xinter, X*Idata2[:,ni]])  #  added to include interaction

    X = np.column_stack([X, Xinter]) #  added to include interaction
    #don't include constant (for fixed effects)
    #X=[Xinter]
    # Create Regressors (NOTE: THE ORDER IS REVERSED, i.e. XX(2) is the first
    # lag in a PVAR(2) and XX(1) is the second lag!!)
    for i in range(lag+1):
        X = np.column_stack([XX[i], X])   
    
    #-----------------------------------------------------------------------------------------------
    # CREATE DEPENDENT VARIABLES
    #----------------------------------------------------------------------------------------------
    Y = [None]*nvar
    for k in range(nvar):
        YY = []
        for j in range(I): 
            tmp = ydata[:,k][lag+j*time:(1+j)*time]
            YY.extend(tmp) 
        Y[k] = YY
        
    #---------------------------------------------------------------------------------------
    # ADJUSTING FOR NaN
    #---------------------------------------------------------------------------------------
    adjsize = int(len(Y[0])/I)
    tempX = None; tempY = None
    positioner = np.zeros((I,2))
    cntryY = [None]*nvar; helpY = [None]*nvar
    
    for kk in range(I):
        for i in  range(nvar): 
            cntryY[i] = Y[i][kk*adjsize:(1+kk)*adjsize]
        cntryX = X[kk*adjsize:adjsize*(1+kk)]
    
        #------------------------------------------------------------------------------------
        # ADJUSTING FOR NaN
        #---------------------------------------------------------------------------------------
        helpY = np.array(cntryY).T
        for j in range(np.shape(cntryX)[1]):
            ROWIND = np.argwhere(~np.isnan(cntryX[:,j]))[:,0]
            newX = []; newY = [];
            for k in ROWIND: 
                newX.append(cntryX[k])
                newY.append(helpY[k])
 
            if len(ROWIND) > 0:
                allnan = 0
                cntryX = np.array(newX)
                helpY = np.array(newY)
            else:
                allnan = 1
                break
    
        if allnan == 1: 
            if kk == 0: 
                positioner[kk,0] = positioner[kk,1] = 0 
            else: 
                positioner[kk,0] = positioner[kk,1] = positioner[kk-1,1] 
            continue
       
        for j in range(np.shape(helpY)[1]):
            ROWIND2 = np.argwhere(~np.isnan(helpY[:,j]))[:,0]
            newX = []; newY = [];
            for k in ROWIND2:
                newX.append(cntryX[k])
                newY.append(helpY[k])
        
            if len(ROWIND2) > 0:
                allnan = 0
                if demean == 1: # if estimations is fixed effect
                    cntryX = newX - np.mean(newX,axis=0)
                    helpY  = newY - np.mean(newY,axis=0)
                else:  # if estimations is not fixed effect
                    cntryX = np.array(newX)
                    helpY = np.array(newY)
        
            else:
                allnan = 1
                break
      
        if allnan == 0:
            if kk == 0:
                positioner[kk,0] = 1 
            else:
                positioner[kk,0] = positioner[kk-1,1] + 1
                
            positioner[kk,1] = positioner[kk,0] + len(cntryX) - 1
            
            if tempY is None:
                tempY = helpY
            else:
                tempY = np.vstack((tempY,helpY))
            if tempX is None:
                tempX = cntryX
            else: 
                tempX = np.vstack((tempX,cntryX))
        ###dummies
        else:
            if kk == 0:
                positioner[kk,0] = positioner[kk,1] = 0 
            else:  
                positioner[kk,0] = positioner[kk,1] = positioner[kk-1,1] 

    Y = np.array(tempY)
    X = np.array(tempX)
    
    # Taking away the lag zero part
    Idata_adj = X[:,:nvar*(Ki+1)]
    X = X[:,nvar*(Ki+1):]
    
    #-----------------------------------------------------------------------------------------------------
    # DEFINE RESTRICTIONS
    #---------------------------------------------------------------------------------------------------
    restr_new = restr
    
    r = [None]*nvar; R = [None]*nvar
    for k in range(nvar):
        if k >= 1:
            n1 = np.sum(restr_new[k,:k*(Ki+1)]) + np.sum(restr_new[k])*lag
            n2 = k*(Ki+1) + (Ki+1)*nvar*lag + 1 + Ki
            r[k] = np.zeros(n1)
            R[k] = np.zeros((n1,n2))
            rowindex = 0
            restrindex = np.kron(np.ones(lag),np.transpose(restr_new[k]))
            for m in range(k):
                restrindex = np.concatenate([restr_new[k,m*(Ki+1):(1+m)*(Ki+1)], restrindex])

            for j in range(len(restrindex)):
                rowindex = int(rowindex+restrindex[j])
                if restrindex[j] == 1:
                    R[k][rowindex-1,j] = 1
               
        elif k == 0:
            n1 = np.sum(restr_new[k,:])*lag
            n2 = (Ki+1)*nvar*lag + 1 + Ki
            r[k] = np.zeros(n1)                
            R[k] = np.zeros((n1,n2))
            rowindex = 0
            restrindex = np.concatenate([np.kron(np.ones(lag),np.transpose(restr_new[k]))])
            for j in range(len(restrindex)):
                rowindex = int(rowindex+restrindex[j])
                if restrindex[j] == 1:
                    R[k][rowindex-1,j] = 1
    
    # Making no interecept variation for the exogenous variables:
    if hominterc > 0:
        for k in range(hominterc):
            r[k] = np.concatenate([r[k], np.zeros(Ki)])
            R[k] = np.vstack([R[k], np.zeros((Ki,np.shape(R[k])[1]))])
            for j in range(Ki):
                R[k][np.shape(R[k])[0]-j-1,np.shape(R[k])[1]-j-1] = 1       
    
    #----------------------------------------------------------------------------------------------------
    # OLS Eq-by-Eq ESTIMATION
    #----------------------------------------------------------------------------------------------------
    # UNRESTRICTED ESTIMATION
    X1 = np.column_stack([X[:,:nvar*lag*(Ki+1)],np.ones(len(X)),X[:,-Ki:]])
    X = X1.copy(); res = [None]*nvar; error = [None]*nvar; errorunres = [None]*nvar
    for k in range(nvar):
        res[k] = np.linalg.solve(np.transpose(X1) @ X1, np.transpose(X1) @ Y[:,k])  
        error[k] = Y[:,k] - X1 @ res[k]
        errorunres[k] = error[k]
        Yh = Idata_adj[:,k*(Ki+1):(1+k)*(Ki+1)] 
        X1 = np.hstack([Yh,  X1])
    
    error = [None]*nvar; variance = [None]*nvar; std = [None]*nvar
    for k in range(nvar):
        tmp  = np.linalg.inv(np.transpose(X) @ X)
        tmp1 = tmp @ np.transpose(R[k])
        tmp2 = np.linalg.inv(R[k] @ tmp1)
        tmp3 = R[k] @ res[k] - r[k]
        res[k] -= tmp1 @ tmp2 @ tmp3  
        error[k] = Y[:,k] - X @ res[k]
        variance[k] = (np.transpose(error[k]) @ error[k])/(len(Y)-nvar*lag-1)   
        std[k]  = np.sqrt(np.diag(tmp) * variance[k])
        Yh = Idata_adj[:,k*(Ki+1):(1+k)*(Ki+1)]
        X = np.hstack([Yh,  X])
        
    #--------------------------------------------------------------------------
    # Testing joint (interaction and variable) Exogeneity (if imposed)
    #--------------------------------------------------------------------------
    numberofres = (1+Ki)*(nvar-1)*lag + 2*Ki
    if hominterc == 0:
        Fvalue = None
    elif hominterc > 0:
       Fvalue = np.zeros(hominterc)
       for i in range(hominterc):
           Fvalue[i] = ((np.transpose(error[i]) @ error[i] - np.transpose(errorunres[i])@errorunres[i]) / numberofres) \
                       * (len(Y[hominterc]-nvar*lag-1)/(np.transpose(errorunres[i]) @ errorunres[i]))
    
    #-----------------------------------------------------------------------------------------------------
    # STORING RESULTS
    #----------------------------------------------------------------------------------------------------
    beta = np.zeros((len(res[nvar-1]),nvar))
    sterr = np.zeros((len(std[nvar-1]),nvar))
    error2 = []
    for k in range(nvar):
        error2.append(error[k])
        beta[len(res[nvar-1])-len(res[k]):,k] = res[k]
        sterr[len(std[nvar-1])-len(std[k]):,k] = std[k]
        
    error2 = np.array(error2).T
    
    return beta,sterr,error2,positioner,Fvalue