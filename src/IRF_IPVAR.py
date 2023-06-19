# Uses Output from the Bootstrap function IPVARboot to create IRFs at
# different evaluations and the respective confidence inrtervals
#
# The inputs are:
# - BETAMAT:  Structural betas from the IPVARboot output
# - ERRORMAT: Structural errors from the IPVARboot output
# - lag:      number of lags
# - values:   Values at which the interaction terms shoud be evaluated
# - period:   The number of periods for which the IRFs should be drawn
# - pct:      The extent of the Confidence interval mistake (i.e. 5 or 10]
# - centered: if 0 normal CIs are drawn if 1 CIs are the (1-pct) range which 
#             minimizes the distance between upper and lower bound
#
# The outputs are:
# - IRF:      The impulse responses
# - STD:      The lower and upper CI of the IRF
# - CUMSTD:   The lower and upper CI for the cummulatieev IRF
# - decomp:   The variance decomposition
# - IMEAN:    The mean of the bootstrapped IRF
# - CUMIMEAN: The mean of the bootstrapped cumulative IRF
#----------------------------------------------------------------------------
# The function calls on: evalpvar.m, irfpreppvar.m, IRFcreator.m, minprctile
# ===========================================================================
# See also IPVARboot, evalpvar, irfpreppvar, IRFcreator, minprctile
#
# Written by Sebastian Weber and Pascal Towbin, 22.01.2009 
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np
from evalpvar import evalpvar
from irfpreppvar import irfpreppvar
from IRFcreator import IRFcreator
from vardecomp import vardecomp
from minprctile import minprctile

def IRF_IPVAR(nint,BETAMAT,ERRORMAT,lag,values,period,pct,centered):
    # ADVANCED BOOTSTRAP METHOD
    
    ne = np.shape(ERRORMAT)[0]
    nvar = np.shape(ERRORMAT)[1]
    neq = nvar
    num_obs = np.shape(BETAMAT)[2]
    b1d = np.ndim(values) == 1
    nv = np.shape(values)[0] if b1d else np.shape(values)[1]
    
    # Allocate spae for arrays
    stdu = np.zeros((period,nvar,nvar,nv))
    stdl = np.zeros((period,nvar,nvar,nv))
    cumstdu = np.zeros((period,nvar,nvar,nv))
    cumstdl = np.zeros((period,nvar,nvar,nv))
    diff_stdu = np.zeros((period,nvar,nvar))
    diff_stdl = np.zeros((period,nvar,nvar))
    diff_cumstdu = np.zeros((period,nvar,nvar))
    diff_cumstdl = np.zeros((period,nvar,nvar))
    STD = np.zeros((period,nvar,nvar,2,nv))
    CUMSTD = np.zeros((period,nvar,nvar,2,nv))
    diff_STD = np.zeros((period,nvar,nvar,2))
    diff_CUMSTD = np.zeros((period,nvar,nvar,2))

    IMEAN    = np.zeros((period,nvar,nvar,nv))
    CUMIMEAN = np.zeros((period,nvar,nvar,nv))
    REDERR = np.zeros((ne,nvar,num_obs,nv))
    IRF = np.zeros((period,nvar,nvar,num_obs,nv))
    decomp = np.zeros((period,nvar,nvar,num_obs,nv))
    
    FPKOM = np.zeros((neq,neq,num_obs))
    for nobs in range(num_obs):
        for i in range(neq):
            FPKOM[i,i,nobs] = np.transpose(ERRORMAT[:,i,nobs]) @ ERRORMAT[:,i,nobs]/((len(ERRORMAT))-(len(np.argwhere(BETAMAT[:,i,0]))))
    
    for val in range(nv):
      for nobs in range(num_obs):
        x = [values[val]] if b1d else values[:,val].tolist()
        BETAEV = evalpvar(nint,BETAMAT[:,:,nobs],x,lag)
        BETIRF, PKOM, AZ = irfpreppvar(BETAEV,FPKOM[:,:,nobs],lag)
        for j in range(len(ERRORMAT)):
            REDERR[j,:,nobs,val] = np.transpose(np.linalg.inv(AZ) @ np.transpose(ERRORMAT[j,:,nobs]))

        tmp = IRFcreator(BETIRF,PKOM,lag,period)
        IRF[:,:,:,nobs,val] = tmp
        tmp = vardecomp(period,IRF[:,:,:,nobs,val])
        decomp[:,:,:,nobs,val] = tmp
    
    STD2 = np.zeros((period,num_obs,nvar,nvar,nv))
    CUMSTD2 = np.zeros((period,num_obs,nvar,nvar,nv))
    for val in range(nv):
      for nobs in range(num_obs):
        for k in range(period): 
            for i in range(nvar): 
                for j in range(nvar): 
                    STD2[k,nobs,i,j,val] = IRF[k,i,j,nobs,val]
        for k in range(period): 
            for i in range(nvar): 
                for j in range(nvar): 
                    CUMSTD2[k,nobs,i,j,val] = np.sum(IRF[:1+k,i,j,nobs,val]) 
    
    # Storing Results
    for val in range(nv):
        for k in range(period):
            for i in range(nvar):
                for j in range(nvar):
                    if centered == 0:
                        stdu[k,i,j,val], stdl[k,i,j,val] = minprctile(STD2[k,:,i,j,val],pct)
                        cumstdu[k,i,j,val], cumstdl[k,i,j,val] = minprctile(CUMSTD2[k,:,i,j,val],pct)
                    elif centered == 1:
                        stdu[k,i,j,val] = np.percentile(STD2[k,:,i,j,val],100-pct/2)
                        stdl[k,i,j,val] = np.percentile(STD2[k,:,i,j,val],pct/2)
                        cumstdu[k,i,j,val] = np.percentile(CUMSTD2[k,:,i,j,val],100-pct/2)
                        cumstdl[k,i,j,val] = np.percentile(CUMSTD2[k,:,i,j,val],pct/2)

                IMEAN[k,i,j,val] = np.mean(STD2[k,:,i,j,val])
                CUMIMEAN[k,i,j,val] = np.mean(CUMSTD2[k,:,i,j,val])
    
    # Storing Difference
    diff_STD2 = STD2[:,:,:,:,0] - STD2[:,:,:,:,1]
    diff_CUMSTD2 = CUMSTD2[:,:,:,:,0] - CUMSTD2[:,:,:,:,1]

    for k in range(period):
        for i in range(nvar):
            for j in range(nvar):
                if centered == 0:
                    diff_stdu[k,i,j], diff_stdl[k,i,j] = minprctile(diff_STD2[k,:,i,j],pct)
                    diff_cumstdu[k,i,j], diff_cumstdl[k,i,j] = minprctile(diff_CUMSTD2[k,:,i,j],pct)
                elif centered == 1:
                    diff_stdu[k,i,j] = np.percentile(diff_STD2[k,:,i,j],100-pct/2)
                    diff_stdl[k,i,j] = np.percentile(diff_STD2[k,:,i,j],pct/2)
                    diff_cumstdu[k,i,j] = np.percentile(diff_CUMSTD2[k,:,i,j],100-pct/2)
                    diff_cumstdl[k,i,j] = np.percentile(diff_CUMSTD2[k,:,i,j],pct/2)

    STD[:,:,:,0,:] = stdl
    STD[:,:,:,1,:] = stdu
    CUMSTD[:,:,:,0,:] = cumstdl
    CUMSTD[:,:,:,1,:] = cumstdu
    
    diff_STD[:,:,:,0] = diff_stdl
    diff_STD[:,:,:,1] = diff_stdu
    diff_CUMSTD[:,:,:,0] = diff_cumstdl
    diff_CUMSTD[:,:,:,1] = diff_cumstdu
    
    return IRF,STD,CUMSTD,decomp,IMEAN,CUMIMEAN,diff_STD,diff_CUMSTD
