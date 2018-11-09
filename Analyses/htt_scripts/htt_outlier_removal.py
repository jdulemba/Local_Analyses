import matplotlib
import matplotlib.pyplot as plt
from pdb import set_trace
import numpy as np
import os, sys
from ROOT import TFile
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--parity')
parser.add_argument('--mass')
parser.add_argument('--width', help='Must be in format 2p5 instead of 2.5')
args = parser.parse_args()

parity, mass, width = args.parity, args.mass, args.width

input_dir = '%s/inputs/htt/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])
output_dir = '%s/results/htt/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])

fname = '%s/%s_%s_%s_limits_gathered.root' % (input_dir, parity, mass, width)
#fname = 'A_600_2p5_limits_gathered'

if not os.path.isfile(fname):
    print 'File with specified parity, mass, and width is missing!'
    sys.exit()

rfile = TFile.Open( fname )
limits = rfile.Get('limit')

expected_cls = np.array( [entry.limit for entry in limits if abs(entry.quantileExpected - 0.5) < 0.0001 if entry.limit > 0.0 if entry.limit != 0.5] )
expected_g = np.array( [entry.g for entry in limits if abs(entry.quantileExpected - 0.5) < 0.0001 if entry.limit > 0.0 if entry.limit != 0.5] )
observed_cls = np.array( [entry.limit for entry in limits if abs(entry.quantileExpected - -1.) < 0.0001 if entry.limit > 0.0 if entry.limit != 0.5] )
observed_g = np.array( [entry.g for entry in limits if abs(entry.quantileExpected - -1.) < 0.0001 if entry.limit > 0.0 if entry.limit != 0.5] )

num_pt_comp = 3 # number of points on either side of point to be used for finding mean, stdev

        ##### try my outlier removal
### deal with CLs close to 0.5
invalid_g = []
invalid_cls = []
invalid_inds = []

lim0p5_idx = np.where( abs(observed_cls-0.5) < 0.05)[0] # indices where cls is close to 0.5
consec = [i for i, df in enumerate(np.diff(lim0p5_idx)) if df!= 1]
consec = np.hstack([-1, consec, len(lim0p5_idx)-1])
consec = np.vstack([consec[:-1]+1, consec[1:]]).T
consec = consec.astype(np.int64)
first_inds = lim0p5_idx[consec][:,0] # first indices of consecutive indices (could be standalone values)
last_inds = lim0p5_idx[consec][:,1] # first indices of consecutive indices (could be standalone values)

# large jump from previous CLs value
cls_jump = [abs(observed_cls[idx]-observed_cls[idx-1]) > 0.2 for idx in first_inds]
#set_trace() 
for j,k in enumerate(cls_jump):
    if k == True:
        for val in range(first_inds[j], last_inds[j]+1):
            invalid_cls.append( observed_cls[val] )
            invalid_g.append( observed_g[val] )
            invalid_inds.append( val )

if invalid_inds:
    print('Points have been removed for being outliers close to CLs = 0.5')
    #set_trace() 

### get rid of invalid indices
observed_inds = [x for x in range(len(observed_cls)) if x not in invalid_inds]
observed_cls = observed_cls[observed_inds]
observed_g = observed_g[observed_inds]
#set_trace() 

z_scores = -1*np.ones(len(observed_cls))
for i in range(len(observed_cls)):
    if i < num_pt_comp: # indices on lower edge of values
            #set_trace()
            bracket = observed_cls[:num_pt_comp+1]
            excluded = bracket[bracket != bracket[i]] # remove value corresponding to current index
            mean, std = excluded.mean(), excluded.std(ddof=1) #uses unbiased estimator for stdev
            z_scores[i] = np.nan if abs((bracket[i] - mean)/std) == np.inf else abs((bracket[i] - mean)/std)

    elif i >= len(observed_cls) - num_pt_comp: # indices on upper edge of values
            bracket = observed_cls[-num_pt_comp-1:]
            excluded = bracket[bracket != bracket[(i+num_pt_comp)-len(observed_cls)]] # remove value corresponding to current index
            mean, std = excluded.mean(), excluded.std(ddof=1) #uses unbiased estimator for stdev
            z_scores[i] = np.nan if abs((bracket[(i+num_pt_comp)-len(observed_cls)] - mean)/std) == np.inf else abs((bracket[(i+num_pt_comp)-len(observed_cls)] - mean)/std)
            #set_trace()

    else: # indices in middle 
            bracket = observed_cls[i-num_pt_comp:i+num_pt_comp+1]
            excluded = np.concatenate((bracket[:num_pt_comp], bracket[-num_pt_comp:]))
            mean, std = excluded.mean(), excluded.std(ddof=1) #uses unbiased estimator for stdev
            z_scores[i] = np.nan if abs((bracket[num_pt_comp] - mean)/std) == np.inf else abs((bracket[num_pt_comp] - mean)/std)
            #if i > 179: set_trace()

    if z_scores[i] > 2.0:
        print('Limit value marked as outlier by z-score, continue, g = ', observed_g[i])
        #set_trace()

#set_trace()
  ## find outlier indices
outliers_inds = np.where(z_scores > 2.0)
outliers_cls = observed_cls[ outliers_inds ]
outliers_g = observed_g[ outliers_inds ]
outliers_z = z_scores[ outliers_inds ]

  ## find valid and non-outlier indices from observed
usable_inds = [x for x in range(len(observed_cls)) if x not in outliers_inds[0]]
usable_cls = observed_cls[ usable_inds ]
usable_g = observed_g[ usable_inds ]
usable_z = z_scores[ usable_inds ]


fig = plt.figure(figsize=[10,10])
fig.suptitle('%s M(%s) %s Limits and z-scores' % (parity, mass, width))

### plot limits
plt.subplot(211)
plt.scatter(expected_g, expected_cls, color='black', label='Expected')
plt.scatter(usable_g, usable_cls, color='green', label='Observed')
if invalid_inds:
    plt.scatter(invalid_g, invalid_cls, color='blue', label='Invalid Points')
if outliers_inds:
    plt.scatter(outliers_g, outliers_cls, color='red', label='Outliers')
plt.grid(which='both')
plt.xlim(0.0, round(observed_g.max()+0.1,2))
plt.ylim(0.001, 1.2)
plt.gca().set_yscale('log')
plt.ylabel('limit')
plt.legend(loc='lower left',fontsize=8, numpoints=1)

### plot z-scores
plt.subplot(212)
plt.scatter(usable_g, usable_z, color='green', label='Observed')
if outliers_inds:
    plt.scatter(outliers_g, outliers_z, color='red', label='Outliers')
plt.grid(which='both')
plt.xlim(0.0, round(observed_g.max()+0.1,2))
plt.ylim(0.001, z_scores[ z_scores > 0. ].max()+1)
plt.gca().set_yscale('log')
plt.xlabel('g')
plt.ylabel('z-score')
plt.legend(loc='lower left',fontsize=8, numpoints=1)
plt.tight_layout()
plt.savefig('%s/%s_M%s_%s_limits_and_zscores.png' % (output_dir, parity, mass, width))
print '\n%s/%s_M%s_%s_limits_and_zscores.png    created' % (output_dir, parity, mass, width)

