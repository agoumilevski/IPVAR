# ============================================================
# Takes the first difference of data which is in panel format.
# Required input data includes:
#  - data: a matrix of dimension (time*I -by- variables)
#  - I:    number of cross sectional units
#
# the output is the first differenced data of dimension
# (time*I -by- variables)
# ============================================================
#
import numpy as np
import pandas as pd

def paneldiff(data,I,concatenate=True):

  chunk = int(len(data)/I)
  output = [None]*I
  output2 = []
  
  for i in range(I):
      x = data[i*chunk:(1+i)*chunk]
      x = np.diff(x,n=1)
      output[i] = [np.nan] + list(x)
      output2 += output[i] 
       
  if concatenate:
      y = pd.DataFrame(np.array(output2),columns=None)
  else:
      y = pd.DataFrame(np.array(output),columns=None)
       
  return y

