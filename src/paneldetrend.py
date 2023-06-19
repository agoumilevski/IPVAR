#
#function [output] =  paneldetrend(data,I, power)
#
# ==========================================================================
# Allows to detrend panel data, according to three different methods defined
# by the input
# - Power:  =0 is demeaning
#           =1 is linear detrending
#           =2 is quadratic detrending
# - I:    Is the cross sectional dimension
# - data: Is the panel data with dimension (time*I - by - variable)
# ==========================================================================
# This code is based on the paper and original code by P. Towbin and S. Weber
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np

def paneldetr(data,I,power):
  
  if np.ndim(data) == 1:
    TT = len(data); K = 1; b1d = True
  else:
    TT,K = np.shape(data); b1d = False
  
  time = TT/I
  if time != np.floor(time):
      print('Error: The time dimension is not identical across countries!!')
  time = int(time)
      
  output = np.zeros(TT)
  for country in range(I):
      start = country*time
      ende = (1+country)*time
      
      trend = np.arange(time)
      ##### Create trend
      X = np.arange(time)
      for i in range(1,1+power):
        X = np.column_stack([X,trend**i])
        
      #a=ifscode(start:ende)
      output1 = None
      for var in range(K):
          if (b1d):
            Y = data[start:ende]
          else:
            Y = data[start:ende,var]
          
          ROWIND  = np.argwhere(~np.isnan(Y))[:,0]
          ROWIND2 = np.argwhere(np.isnan(Y))[:,0]
          newY = np.zeros(len(ROWIND))
          if b1d:
              newX = np.zeros(len(ROWIND))
          else:
              newX = np.zeros((len(ROWIND),np.shape(X)[1]))
          # Remove np.nan for estimations
          if (len(ROWIND) > 0):
              for k in range(len(ROWIND)):
                 if (b1d):
                   newY[k] = Y[ROWIND[k]]
                   newX[k] = X[ROWIND[k]]
                 else:
                   newY[k] = Y[ROWIND[k]]
                   newX[k] = X[ROWIND[k]]
                 
              xx = np.transpose(newX) @ newX
              xy = np.transpose(newX) @ newY
              if b1d:
                  BETAA = xy/xx
                  detrend_raw = newY - newX * BETAA
              else:
                  BETAA = np.linalg.solve(xx,xy)
                  detrend_raw = newY - newX @ BETAA
              # refill np.nan to assure uniform data len
              detrended = np.empty(len(Y))
              detrended[:] = np.nan
              detrended[ROWIND] = detrend_raw
              
          else:
            detrended = Y
          
          if output1 is None:
              output1 = detrended
          else:
              output1 = np.column_stack([output1, detrended])
        
      
      output[start:ende] = output1
      
  
  return output


