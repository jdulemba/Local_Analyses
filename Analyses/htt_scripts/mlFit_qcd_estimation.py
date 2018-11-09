import numpy as np
from scipy.optimize import minimize
from scipy.misc import factorial
import matplotlib.pyplot as plt
from pdb import set_trace
from argparse import ArgumentParser
import Utilities.python.prettyjson as prettyjson
import os

parser = ArgumentParser()
parser.add_argument('--lepton', help='Select between muons or electrons')
parser.add_argument('--njets', help='Select between 3 or 4+ jet categories')

args = parser.parse_args()

input_dir = '%s/inputs/htt/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])
output_dir = '%s/results/htt/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])

"""
created by Joseph Dulemba on September 14 2018

This script is to estimate the qcd contribution in the signal region
using info from section 8.1 of AN-2016/272.

"""


##Contribution from data in regions A (tight/MTHigh), B (looseNOTTight/MTHigh), C (tight/MTLow), and D (looseNOTTight/MTLow)
##Values of data, non-QCD MC (prompt), and QCD MC in lists of [N_A, N_B, N_C, N_D], except QCD (0 used for N_A)
regions = ['tight/MTHigh', 'looseNOTTight/MTHigh', 'tight/MTLow', 'looseNOTTight/MTLow']



### define negative log-likelihood function
def negLogLikelihood(params, N_data, N_prompt, N_QCD):
    """ the negative log-Likelihood-Function"""
    scale, N_QCD_A = params

    lnl_A = -1*N_data[0]*np.log(scale*N_prompt[0]+N_QCD_A)+(scale*N_prompt[0]+N_QCD_A)
    lnl_B = -1*N_data[1]*np.log(scale*N_prompt[1]+N_QCD[1])+(scale*N_prompt[1]+N_QCD[1])
    lnl_C = -1*N_data[2]*np.log(scale*N_prompt[2]+N_QCD[2])+(scale*N_prompt[2]+N_QCD[2])
    lnl_D = -1*N_data[3]*np.log(scale*N_prompt[3]+((N_QCD[1]*N_QCD[2])/N_QCD_A))+(scale*N_prompt[3]+((N_QCD[1]*N_QCD[2])/N_QCD_A))
    #set_trace()
    lnl = np.sum(lnl_A+lnl_B+lnl_C+lnl_D)

    return lnl




def indiv_yield( lepton, njets):
    ##open yields file corresponding to lepton type and number of jets
    fname = '%s_3Jets_yields.json' % lepton if njets == '3' else '%s_4PJets_yields.json' % lepton
    yields = prettyjson.loads( open( '%s/%s' % (input_dir, fname) ).read() )
    
    
    ## create lists of values for data, prompt, and QCD for each region
    N_data, N_prompt, N_QCD = np.zeros(4).tolist(), np.zeros(4).tolist(), np.zeros(4).tolist()
    
    for i,j in enumerate(regions):
        N_data[i], N_prompt[i] = yields[j]['Observed'][0], yields[j]['Prompt'][0]
    
    N_QCD = [N_data[i]-N_prompt[i] for i in range(len(N_data))]


    ### minimize the negative log-Likelihood
    init_point = [0, 2000]
    result = minimize(negLogLikelihood,  # function to minimize
                      x0=init_point,     # start value, params
                      args=(N_data, N_prompt, N_QCD),      # additional arguments for function
                      method='Powell',   # minimization method, see docs
                      #method='CG',   # minimization method, see docs
                      #method='Nelder-Mead',   # minimization method, see docs
                      )
    # result is a scipy optimize result object, the fit parameters 
    # are stored in result.x
    #print(result)
    return result.x[1]

if args.lepton and args.njets:
    ml_yield = indiv_yield(args.lepton, args.njets)

else:
    ml_yields = {}
    for lep in ['muons', 'electrons']:
	ml_yields[lep] = {}
        for njet in ['3', '4+']:
	    ml_yield = indiv_yield(lep, njet)
	    ml_yields[lep][njet] = ml_yield

    #set_trace()
    res_fname = 'mlFit_yields.json'
    with open( '%s/%s' %(output_dir, res_fname), 'w' ) as f:
	f.write(prettyjson.dumps(ml_yields))    

    print '\n-----   resulting QCD yields written to %s/%s   -----\n' % (output_dir, res_fname)


### attempt to estimate errors from minimization (currently dosen't work)
#def Reliabilities():
#    points = 1000
#    scale = np.linspace( result.x[0]-result.x[0]/10, result.x[0]+result.x[0]/10, points) # +- 10% of result for scale
#    scale_spacing = scale[1]-scale[0]
#    N_QCD_A = np.linspace( int(result.x[1])-(int(result.x[1])/2), int(result.x[1])+(int(result.x[1])/2), points) # +- 50% of result for N_QCD_A
#    N_QCD_A_spacing = N_QCD_A[1]-N_QCD_A[0]
#    neglnL = np.zeros((points,points))
#
#    # Joint PDF
#    for i in range(points):
#        for j in range(points):
#            lnl_A = -1*N_data[0]*np.log(scale[i]*N_prompt[0]+N_QCD_A[j])+(scale[i]*N_prompt[0]+N_QCD_A[j])
#            lnl_B = -1*N_data[1]*np.log(scale[i]*N_prompt[1]+N_QCD[1])+(scale[i]*N_prompt[1]+N_QCD[1])
#            lnl_C = -1*N_data[2]*np.log(scale[i]*N_prompt[2]+N_QCD[2])+(scale[i]*N_prompt[2]+N_QCD[2])
#            lnl_D = -1*N_data[3]*np.log(scale[i]*N_prompt[3]+((N_QCD[1]*N_QCD[2])/N_QCD_A[j]))+(scale[i]*N_prompt[3]+((N_QCD[1]*N_QCD[2])/N_QCD_A[j]))
#            #set_trace()
#            neglnL[i,j] = np.sum(lnl_A+lnl_B+lnl_C+lnl_D)
#
#    #set_trace()
#    ## find indices for neg logL minimum
#    #min_indices = np.argwhere(neglnL == np.min(neglnL))[0]
#
#    set_trace()
#    ## Marginal PDFs
#    #Marg_scale = np.sum(np.exp(-neglnL), axis = 0)
#    ##print(Marg_x)
#    #Marg_N_QCD_A = np.sum(np.exp(-neglnL), axis = 1)
#
#    #ln_N_QCD_A = np.log(Marg_N_QCD_A)
#    #N_QCD_A_bin_max = Marg_N_QCD_A.argmax()
#    #d2_N_QCD_A = (ln_N_QCD_A[N_QCD_A_bin_max+1]-2*ln_N_QCD_A[N_QCD_A_bin_max]+ln_N_QCD_A[N_QCD_A_bin_max-1])/(N_QCD_A_spacing**2)
#    #rel_N_QCD_A = (-d2_N_QCD_A)**(-0.5)
#
#    # Reliability
#    ln_N_QCD_A = np.sum(-neglnL, axis = 1)
#    N_QCD_A_bin_max = ln_N_QCD_A.argmax()
#    d2_N_QCD_A = (N_QCD_A[N_QCD_A_bin_max+1]-2*N_QCD_A[N_QCD_A_bin_max]+N_QCD_A[N_QCD_A_bin_max-1])/(N_QCD_A_spacing**2)
#    #d2_N_QCD_A = (ln_N_QCD_A[N_QCD_A_bin_max+1]-2*ln_N_QCD_A[N_QCD_A_bin_max]+ln_N_QCD_A[N_QCD_A_bin_max-1])/(N_QCD_A_spacing**2)
#    rel_N_QCD_A = (-d2_N_QCD_A)**(-0.5)
#    #kmax = np.exp(x[x_bin_max]) # best estimate of k
#    #k1sig_min = np.exp(x[x_bin_max]-rel_x) # min value of k moving 1SD
#    #k1sig_max = np.exp(x[x_bin_max]+rel_x) # max value of k moving 1SD
#    #k1sig_avg = (np.abs(kmax-k1sig_min)+np.abs(kmax-k1sig_max))/2 # average of k value differences
#
#
##Reliabilities()
##set_trace()
