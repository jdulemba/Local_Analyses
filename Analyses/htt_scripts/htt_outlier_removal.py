import matplotlib
import matplotlib.pyplot as plt
import uproot
import pandas as pd
import numpy as np
from pdb import set_trace

tf = uproot.open('A_600_2p5_limits_gathered.root')
#tf = uproot.open('A_550_5_limits_gathered.root')
df = pd.DataFrame(tf['limit'].arrays(tf['limit'].keys()))
expected = df[np.abs(df[b'quantileExpected'] - 0.5 ) < 0.01][(df[b'limit'] > 0.) & (df[b'limit'] != 0.5)]
observed = df[np.abs(df[b'quantileExpected'] - -1. ) < 0.01][(df[b'limit'] > 0.) & (df[b'limit'] != 0.5)]

num_pt_comp = 3 # number of points on either side of point to be used for finding mean, stdev
z_scores = -1*np.ones(len(observed))

for i in range(len(observed)):
	if i < num_pt_comp:
		#set_trace()
		bracket = observed[:num_pt_comp+1][b'limit']
		excluded = bracket[bracket != bracket.iloc[i]] # remove value corresponding to current index
		mean, std = excluded.mean(), excluded.std()
		z_scores[i] = np.nan if abs((bracket.iloc[i] - mean)/std) == np.inf else abs((bracket.iloc[i] - mean)/std)

	elif i >= len(observed) - num_pt_comp:
		bracket = observed[-num_pt_comp-1:][b'limit']
		excluded = bracket[bracket != bracket.iloc[(i+num_pt_comp)-len(observed)]] # remove value corresponding to current index
		mean, std = excluded.mean(), excluded.std()
		z_scores[i] = np.nan if abs((bracket.iloc[(i+num_pt_comp)-len(observed)] - mean)/std) == np.inf else abs((bracket.iloc[(i+num_pt_comp)-len(observed)] - mean)/std)
		#set_trace()

	else:
		bracket = observed[i-num_pt_comp:i+num_pt_comp+1][b'limit']
		excluded = pd.concat((bracket[:num_pt_comp], bracket[-num_pt_comp:]))
		mean, std = excluded.mean(), excluded.std()
		z_scores[i] = np.nan if abs((bracket.iloc[num_pt_comp] - mean)/std) == np.inf else abs((bracket.iloc[num_pt_comp] - mean)/std)
		#if i > 179: set_trace()

print(z_scores[ z_scores > 2.0]) 

plt.figure(figsize=[10,10])
plt.subplot(211)
plt.scatter(expected[b'g'], expected[b'limit'])
plt.scatter(observed[b'g'], observed[b'limit'])
plt.grid(which='both')
plt.xlim(0.0, 3.)
plt.ylim(0.001, 1.2)
plt.gca().set_yscale('log')
plt.ylabel('limit')
plt.subplot(212)
plt.scatter(observed[b'g'],z_scores)
plt.grid(which='both')
plt.xlim(0.0, 3.)
plt.ylim(0.001, z_scores[ z_scores > 0. ].max()+1)
plt.gca().set_yscale('log')
plt.xlabel('g')
plt.ylabel('z-score')
plt.tight_layout()
plt.show()
set_trace()
