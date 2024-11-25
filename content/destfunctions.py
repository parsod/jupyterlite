import numpy as np
from numpy.matlib import repmat,randn

def createmcdata(yinp,ysiginp,nmc,distrib):

	#      function ymc = createmcdata(y,ysig,nmc,distrib)
	#
	# Creates a matrix ymc of nmc vectors with the mean values of y but with
	# added random noise of standard deviation ysig. 
	#
	#     y       data vector
	#     ysig    standard deviation vector (same length as y)
	#     nmc     number of Monte Carlo copies
	#     distrib 'norm' gives normal distribution
	#             'lognorm' give lognormal distribution (useful for example 
	#             if negative results are unphysical)
	#
	#
	#  You might want to initialize the random number generator in forehand.
	#

	if np.size(yinp) != np.size(ysiginp):
		raise Exception('y and ysig must be vectors of the same length.')

	n=np.size(yinp)
	y=yinp.reshape((1,n))
	ysig=ysiginp.reshape((1,n))
	if distrib.lower() in ('norm' ,'normal'):
		
		return np.array(repmat(y,nmc,1)) + np.array(repmat(ysig,nmc,1))*np.array(randn(nmc,n))
	elif  distrib.lower() in ('lognorm','lognormal'):
			mu = np.log(y**2/np.sqrt(ysig**2+y**2))  # mu of lognormal dist
			sigma = np.sqrt(np.log(ysig**2/y**2+1))  # sigma of lognormal dist
			return np.exp(np.array(randn(nmc,n))*np.array(repmat(sigma,nmc,1)) + np.array(repmat(mu,nmc,1)))
	else:
		raise Exception('Distribution named "' + distrib + '" is not recognized.')


def xcyclo(n):
   #Returnerar molbråket cyklohexan givet brytningsindex för en blandning av
   #cyklohexan-etanol vid 25 grader.
    p =  1.0e+03 * np.array([1.043827711247225 , -4.274100703396606 ,  5.845780159402767 , -2.670582627220946])
    return np.polyval(p,n)

def mcpolyfit(xinp, yinp, n, level):
    #Performs polynomial fitting p(x)=y with error analysis 
    #using a Monte Carlo approach.
    
    #Input arguments:
    #  x : a NX x N matrix: the NX data sets of x values (N data points)
    #  y : a NY x N matrix: the NY data sets of y values (N data points)
    #      NX and NY need not be the same. In particular one may use a
    #      single data set (without added noise) for one of them.
    #      The number of fits equals max(NX,NY) and if there are less data
    #      sets for one of x or y, they are just cyclically reused.
    #  n : the polynomial degree used for fitting (1 for linear regression)
    #  level : the confidence level for the returned confidence interval
    #          (must be between 0.5 and 1.0)
    #Return values:
    #  Returns four arrays, each of length n+1, containing the
    #  statistical analysis of the distribution of the fitted
    #  parameters:
    #  pmean : average value of each parameter (can be used as the result)
    #  psig  : standard deviation of each parameter
    #  plow  : low confidence bound for each parameter
    #  phigh : high confidence bound for each parameter
    
    if np.ndim(xinp) == 1:
        x=xinp.reshape((1,np.size(xinp)))
    else:
        x= xinp
    if np.ndim(yinp) == 1:
        y.yinp.reshape((1,np.size(yinp)))
    else:
        y=yinp
    if np.size(x,1) != np.size(y,1):
        raise Exception('Number of columns in x and y must be equal')
    if level>1 or level<0.5:
        raise Exception('Illegal confidence level')

    xn=np.size(x,0)
    yn=np.size(y,0)
    nmc = max(xn,yn)
    pmc = np.zeros((nmc,n+1)) 
    for i in range(nmc):
        pmc[i,:]=np.polyfit(x[i%xn,:],y[i%yn,:], n)
    #Statistisk analys av parametrarna i pmc: 
    # medelvärde, standardavvikelse och konfidensintervall
    #(t.ex. första och sista ventilen om nivån är 95%)
    pmean = np.mean(pmc,0)
    psig = np.std(pmc,0)
    plow = np.zeros(n+1)
    phigh = np.zeros(n+1)
    for j in range(n+1):
        tmp=np.sort(pmc[:,j])
        plow[j]=tmp[round(max(1,0.5*(1-level)*nmc))-1]
        phigh[j]=tmp[round(min(nmc,1-0.5*(1-level)*nmc))-1]
    return (pmean,psig,plow,phigh)
	
	
def mcpolyfitdemo(xinp, yinp, n):
	#This function is for demonstration purposes only, it's not needed for the analysis.
	#Arguments aare the same as for mcpolyfit
    
    if np.ndim(xinp) == 1:
        x=xinp.reshape((1,np.size(xinp)))
    else:
        x= xinp
    if np.ndim(yinp) == 1:
        y.yinp.reshape((1,np.size(yinp)))
    else:
        y=yinp
    if np.size(x,1) != np.size(y,1):
        raise Exception('Number of columns in x and y must be equal')

    xn=np.size(x,0)
    yn=np.size(y,0)
    nmc = max(xn,yn)
    pmc = np.zeros((nmc,n+1)) 
    for i in range(nmc):
        pmc[i,:]=np.polyfit(x[i%xn,:],y[i%yn,:], n)
    return pmc
