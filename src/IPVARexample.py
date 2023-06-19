# ==========================================================================================
# Interacted Panel VAR model.
#
# This code is based on the work of Pascal Towbin and Sebastian Weber
# "Limits of floating exchange rates: The role of foreign currency debt and import structure",
# Journal of Development Economics, 2013, vol. 101, issue C, 179-194
# ==========================================================================================

import os
import numpy as np
import pandas as pd
from time import time
from paneldiff import paneldiff
from interactPVAR import interactPVAR
from evalpvar import evalpvar
from irfpreppvar import irfpreppvar
from IRFcreator import IRFcreator
from IPVARboot import IPVARboot
from IRF_IPVAR import IRF_IPVAR
from util import plotCharts

## SET WORKING DIRECTORY
fpath = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(fpath,"../"))
os.chdir(working_dir)

t0 = time()
## RETRIEVE DATA 
data = pd.read_csv(os.path.join(working_dir,"data/data.csv"))
data = data.apply(pd.to_numeric, errors='coerce')

## Define Variables
tot      = data["tot"].values
inv      = data["inv"].values
gdp      = data["gdp"].values
debt     = data["debt"].values
exrate   = data["exrate"].values
ifscode  = data["ifscode"].values
year     = data["year"].values

################################### Setting Estimation Parameters ###########################################
lag = 2        # Number of lags
hominterc = 1  # in the first "hominterc" the interaction terms have no level effect
               #  (useful for block exogenous variables with homogeneous dynamics and intercepts)
I = 109        # Number of countries
pct = 10       # Confidence level for the CI-bands (default = 10)
nsim = 5     # Number of simulations for the bootstrapped confidence bands (default = 200)
centered = 0   # If = 0 symmetric CIs are reported if = 1 CIs are the (1-pct) range
               #  which minimizes the distance between upper and lower bound (default =0)
parametric = 1 # If = 1 CI are based on drawings of standards errors from normal
               #  distribution, otherwise (0) - from observed errors (default = 1)
period = 11    # Number of periods for which IRF is calculated (default = 10)
demean = 1     # 1 = fixed effects (default = 1)
cum = 0        # Cumulative IRFs = 1 (default = 1)
shock = 0      # Graph IRF in response to shock in variable # 1,2,3....nvar
#############################################################################################################

## Define regressors
# first differences of the log of the data while taking account of the panel nature of the data
ydata = np.hstack([paneldiff(np.log(100*tot),I).values,paneldiff(inv,I).values,paneldiff(gdp,I).values])

variables = ["Terms of Tade","Investment","GDP"]
impulse = "IR"

## Define Matrix of Interaction Terms
Idata = np.column_stack([exrate,debt,exrate*debt])

nint = np.shape(Idata)[1] # number of interaction terms
nvar = np.shape(ydata)[1] # number of endogenous variables

## Setting Restrictions on Interaction Terms
# - restr:   Is a matrix of size nvar:(nint+1)*nvar with zeros and ones defining for
#            which equation which interaction and variable a restriction is supposed
#            to be applied (ones leading to an exclusion of the interaction or variable).
# Eight equations, two interaction variables and, first variable exogenous.

r0 = [0]*(1+nint); r1 = [0]+[1]*nint; r2 = [1]*(1+nint)
restr = [ r1, r2, r2, 
          r0, r0, r0, 
          r0, r0, r0]
restr = np.array(np.concatenate(restr))
restr = np.reshape(restr,newshape=[nvar,(1+nint)*nvar])
            
## See interactPVAR for a description for no restrictions on dynamics set
## Defining values at which interaction terms are to be evaluated for the IRF
idebt = debt[~np.isnan(debt)]
q1 = np.quantile(idebt,q=0.75) 
q2 = np.quantile(idebt,q=0.25)
values=[1, q1, q1,
        0, q1, 0,
   	    1, q2, q2,
	    0, q2, 0];
values = np.reshape(values,newshape=[1+nint,nvar])
values = np.transpose(values)

b1d = np.ndim(values)==1
nv = len(values) if b1d else np.shape(values)[1]

## Estimation of structural form
beta,sterr,errors,positioner,Fvalue = interactPVAR(ydata,I,lag,Idata,hominterc,restr,demean)

## Variance of (orthogonal) shocks
fPkomega = np.zeros((nvar,nvar))
for i in range(nvar):
    fPkomega[i,i] = np.transpose(errors[:,i]) @ errors[:,i]/(len(errors)-np.count_nonzero(beta[:,i]))

## Create IRF for different interaction term values
betaeval = list(); MA = list()
for i in range(nv):
    v = [values[i]] if b1d else values[:,i].tolist()
    tmp = evalpvar(nint,beta,v,lag)
    betairf,Pkomega,_ = irfpreppvar(tmp,fPkomega,lag)
    betaeval.append(betairf)
    tmp = IRFcreator(betairf,Pkomega,lag,period)
    MA.append(tmp)

# Create bootstrapped structural betas
print("\nBootstrapping...\n")
BETAMAT,ERRORMAT,STERRMAT = IPVARboot(parametric,positioner,Idata,ydata,errors,beta,nsim,I,lag,restr,hominterc,demean)

# Create bootstrapped IRFs
IRF,STD,CUMSTD,decomp,IMEAN,CUMIMEAN,diff_STD,diff_CUMSTD = IRF_IPVAR(nint,BETAMAT,ERRORMAT,lag,values,period,pct,centered)

## Plot charts
print("Plotting charts...")
plotCharts(MA,STD,CUMSTD,diff_STD,diff_CUMSTD,shock,cum,period,nv,impulse,variables,showDifference=False,save=True)

elapsed = time() - t0
print("\nElapsed time: %.1f (seconds)" % elapsed)
print("\nDone!")