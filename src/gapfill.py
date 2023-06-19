# ==========================================================================
# Interpolates data and adds data at the beginning and end of the vector if
# there are NaNs
#
# For instance
# If only internal values are supposed to be interpolated use
# Y = gapfill(X, 'method')
#
# If also the endings should be interpolated use
# Y = gapfill(X, 'method1', 'method2')
#
# where method 1 is the method applied to the internal values and method2
# is the method applicable to the missing values at the end and beginning
# of the vector. Standard methods include 'linear'.
# See 'interp1' for a description of possible methods
# =======================================================================
# This code is based on the paper and original code by P. Towbin and S. Weber
# Translated to Python by A. Goumilevski - 05.01.2023


import numpy as np
from scipy.interpolate import interp1d

def gapfill(X,method1=None,method2=None):

  if method1 is None and method2 is None:
      print('Error: Please specify an interpolation method!')
      Y = list()
      return Y
    
  time = np.arange(len(X))
  Y = X.copy()
  X0 = X[~np.isnan(X)]
  existx = time[~np.isnan(X)]
  missingx = time[np.isnan(X)]
  
  func = interp1d(x=existx,y=X0,kind=method1)
  try:
      yi = func(missingx)
      for i in range(len(missingx)):
          Y[missingx[i]] = yi[i]
  except:
      pass
      
  if not method2 is None:
    Y0 = Y[~np.isnan(Y)]
    existY = time[~np.isnan(Y)]
    missingY = time[np.isnan(Y)]
    func = interp1d(x=existY,y=Y0,kind=method2,fill_value="extrapolate")
    yi2 = func(missingY)
    for i in range(len(missingY)): 
       Y[missingY[i]] = yi2[i]
      
  return Y
