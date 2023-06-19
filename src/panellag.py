#
#function [output] =  panellag(data,I,length);
#
# ==================================================================
# Lags the whole Panel data by the wished number of lags.
# Required inputs are
#  - data:   The panel data with dimenstion (time*I - by- variables)
#  - I:      number of cross sectional units
#  - length: the number of lags that should be taken from the data
#
# Output
#  - output: a matrix of dimension (time*I - by- variables) with the
#            lagged data
# ===================================================================
# See also paneldiff, paneldetrend
#
# Based on the Matlab code of Pascal Towbin and Sebastian Weber
# Translated to Python by A. Goumilevski - 05.01.2023
#
import numpy as np

def panellag(data,I,length):

  time = len(data)/I
  #K = len(data)/time
  if (time != np.floor(time)):
      print('Error: The time dimension is not identical across countries!!')
  time = int(time)
    
  output = np.zeros(I*time)
  for actual_country in range(I):
        start = actual_country*time
        ende = (1+actual_country)*time
        output[start:(start+length)] = np.nan
        output[(length+start):ende] = data[start:(ende-length)]
  

  return(output)


