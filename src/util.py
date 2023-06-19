import os
import numpy as np
import matplotlib.pyplot as plt
    
fpath = os.path.dirname(os.path.abspath(__file__))
j = 0 

def chart(MA,STD,diff_STD,numpic,nvar,nv,shock,impulse,variables,showDifference=True):
    #MA is period x Number of IRs matrix and stands for the IRS that should be comparded in a row 
    #STD is a period x 2 matrix Number of IRs   and contains the confidence intervals 
    
    # GENERATE IRF plots
    #-------------------------------------------------------------------
    
    # Number of periods and IRS
    global j
    period = np.shape(MA)[0]
    
    if showDifference:
        nc = 2*nv-1
        if nv == 2:
            nc = 3
            titles = ["High Debt","Low Debt","Difference"]
        else:
            titles = []
            for i in range(nv):
                titles.append(f"Interaction: {1+i}")
                if i > 0:
                    titles.append("Difference")  
    else:
        nc = nv
        titles = [f"Interaction values: Choice {1+i}" for i in range(nv)]
    
    MA = 100*MA; STD = 100*STD; diff_STD = 100*diff_STD
    diffMA = np.transpose(np.array([MA[:,0] - MA[:,i] for i in range(1,nv)]))
    zeros = np.zeros(period)
    
    mult = 1.1
    max1 = np.max(STD); min1 = np.min(STD)
    maximum = max1*np.sign(max1)*mult+max1*(1-np.sign(max1))/mult
    minimum = min1*(1-np.sign(min1))*mult+min1*(np.sign(min1))/mult
    
    #-------------------------------------------------------------------
    # PLOT IRFs
    #-------------------------------------------------------------------
    i1 = -1; i2 = -1
    for i in range(nc):
        j += 1
        plt.subplot(nvar,nc,j)
        plt.plot(zeros,color='k',linestyle='solid',linewidth=1)
        if showDifference:
            if i<2 or i%2==1:
                i1 += 1
                plt.plot(MA[:,i1],color='red',linestyle='solid',linewidth=2)
                plt.plot(STD[:,0,i1],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
                plt.plot(STD[:,1,i1],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
                plt.fill_between(range(period),STD[:,0,i1],STD[:,1,i1],color='blue',alpha=0.1)               
            else:
                i2 += 1
                plt.plot(diffMA[:,i2],color='red',linestyle='solid',linewidth=2)
                plt.plot(diff_STD[:,0,i2],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
                plt.plot(diff_STD[:,1,i2],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
                plt.fill_between(range(period),diff_STD[:,0,i2],diff_STD[:,1,i2],color='blue',alpha=0.1)
        else:
            plt.plot(MA[:,i],color='red',linestyle='solid',linewidth=2)
            plt.plot(STD[:,0,i],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
            plt.plot(STD[:,1,i],color='blue',linestyle='dashed',dashes=(10,10),linewidth=0.5)
            plt.fill_between(range(period),STD[:,0,i],STD[:,1,i],color='blue',alpha=0.1)
            
        plt.grid(True)
        plt.xlim((0,period-1))
        if numpic == nvar-1:
            plt.xlabel("Quarter",fontsize=15)
        else:
            plt.gca().axes.xaxis.set_ticklabels([])
            
        if not showDifference: 
            plt.ylim((minimum,maximum))
    
        if i == 0:
            plt.ylabel(variables[numpic],fontsize=15)
    
        if j <= nc:
            plt.title(titles[i],size=20 if len(titles[0]) < 10 else 15,fontweight="bold")

        
def plotCharts(MA,STD,CUMSTD,diff_STD,diff_CUMSTD,shock,cum,period,nv,impulse,variables,showDifference=True,save=False):
    # Creates (cumulative) IRFs ordered by values at which the interactions are
    # supposed to be evaluated.
    nvar = len(variables)
    IR = np.zeros((period,nv))
    CR = np.zeros((period,2,nv)) 
    diff_CR = np.zeros((period,2,nv)) 
    
    plt.figure(figsize=(20,15)) 
    
    for numpic in range(nvar):
        if cum == 1:
            for r in range(nv):
                normalizer = np.sum(MA[r][:,shock,shock])
                IR[:,r] = np.cumsum(MA[r][:,numpic,shock]/normalizer)
                CR[:,0,r] = CUMSTD[:,numpic,shock,0,r]/normalizer
                CR[:,1,r] = CUMSTD[:,numpic,shock,1,r]/normalizer
                diff_CR[:,0,r] = diff_CUMSTD[:,numpic,shock,0]/normalizer
                diff_CR[:,1,r] = diff_CUMSTD[:,numpic,shock,1]/normalizer
            chart(IR,CR,diff_CR,numpic,nvar,nv,shock,impulse,variables,showDifference=showDifference) 
        elif cum == 0:   
            for r in range(nv):
                normalizer = MA[r][0,shock,shock]
                IR[:,r] = MA[r][:,numpic,shock]/normalizer
                CR[:,0,r] = STD[:,numpic,shock,0,r]/normalizer
                CR[:,1,r] = STD[:,numpic,shock,1,r]/normalizer
                diff_CR[:,0,r] = diff_STD[:,numpic,shock,0]/normalizer
                diff_CR[:,1,r] = diff_STD[:,numpic,shock,1]/normalizer
            chart(IR,CR,diff_CR,numpic,nvar,nv,shock,impulse,variables,showDifference=showDifference)

    if save:
        plt.savefig(os.path.join(fpath,"../graphs/ipvar.png"), dpi=300)
    plt.show()

