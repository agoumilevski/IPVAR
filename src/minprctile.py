# =============================================================================
# Returns the upper and lower value of the values in X which define the
# (1-pctile) perecent of the values. Rather than taking the 2.5# and 97.5
# limits in the case of pctile=5, the values L and U are chosen to minimize
# the distance of U and L mantainig the area between U and L equal to 1-pctile.
# -----------------------------------------------------------------------------
# Calls on the function prctile
# =============================================================================
#
# This code is based on the paper and the original code by P. Towbin and S. Weber
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np

def minprctile(X,pctile):

  # "exact" is the decimeter unit at which the minimization is to be done
  # the smaller it is chosen: the finer the grid to find the minimizing area.
  exact  = 0.1
  length = int(pctile/exact)
  
  lower = np.zeros(length+1)
  upper = np.zeros(length+1)
  
  for i in range(length+1):
      pctile2 = exact*i
      lower[i] = np.quantile(X,(pctile2/100))
      upper[i] = np.quantile(X,(100-pctile+pctile2)/100)
  
  dist  = upper - lower
  I = np.argmin(dist)
  L = lower[I]
  U = upper[I]
  
  return U,L
