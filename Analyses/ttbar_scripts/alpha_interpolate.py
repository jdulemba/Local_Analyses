import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pdb import set_trace
import pickle
import Utilities.python.prettyjson as prettyjson

## import prettyjson file from parent dir
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)
#import prettyjson as prettyjson

input_dir = '%s/inputs/ttbar/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])
output_dir = '%s/results/ttbar/%s' % (os.environ['ANALYSES_PROJECT'], os.environ['jobid'])

fname = '%s/median_Mtt_Mthad.json' % input_dir
median_dict = prettyjson.loads(open(fname).read())

mtt_ranges_to_centers = {
	'M($t\\overline{t}$) 200-350' : 275, 'M($t\\overline{t}$) 350-400' : 375, 'M($t\\overline{t}$) 400-500' : 450,
	'M($t\\overline{t}$) 500-700' : 600, 'M($t\\overline{t}$) 700-1000' : 850, 'M($t\\overline{t}$) $\geq$1000' : 1200
}
mthad_ranges_to_centers = {
	'M($t_{h}$) 0.9-1.1' : 1.0, 'M($t_{h}$) 1.1-1.3' : 1.2, 'M($t_{h}$) 1.3-1.5' : 1.4, 'M($t_{h}$) 1.5-1.7' : 1.6,
	'M($t_{h}$) 1.7-1.9' : 1.8, 'M($t_{h}$) 1.9-2.1' : 2.0, 'M($t_{h}$) 2.1-2.3' : 2.2, 'M($t_{h}$) 2.3-2.5' : 2.4
}

mttbar_ranges = [ 'Mttbar200to350', 'Mttbar350to400', 'Mttbar400to500', 'Mttbar500to700', 'Mttbar700to1000', 'Mttbar1000toInf']

mthad_bin_edges = np.arange(0.9, 2.7, 0.2).tolist()
mttbar_bin_edges = [200, 350, 400, 500, 700, 1000]
alpha_bin_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

colors = ['green', 'red', 'black', 'blue', 'orange', 'magenta', 'cyan', 'yellow']

### create grid of ( x = 173.1/Mthad, y = mttbar, z = medians )

x = np.array([1.0 , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4]) # 173.1/mthad vals
y = np.array([275, 375, 450, 600, 850, 1200]) # mttbar bins

for alpha_key in median_dict.keys():
	z = np.array([median_dict[alpha_key][mtt_range]['medians'] for mtt_range in mttbar_ranges]) # median alpha values
	
	f = interpolate.interp2d(x, y, z, kind='cubic')

		## save interpolation function for each alpha
	with open('%s/Alpha_%s_Interpolation.pkl' % (output_dir, alpha_key), 'wb') as pkl_file:
		pickle.dump(f, pkl_file)

	#interp_loaded = 0
	#with open('%s/Alpha_%s_Interpolation.pkl' % (output_dir, alpha_key), 'rb') as interpolate:
	#	interp_loaded = pickle.load(interpolate)

	xnew = np.linspace(0.9,3.0, 500)
	z500 = f(xnew, 500) ## estimate new alphas for mttbar = 500
	z350 = f(xnew, 350) ## estimate new alphas for mttbar = 350
	
	## make alpha vs 173.1/MTHad plots
	fig, ax = plt.subplots()
	for i in range(len(y)):
		leg_label = [ k for k,v in mtt_ranges_to_centers.items() if v == y[i] ][0]
		plt.plot(x, z[i,:], linestyle='None', color=colors[i], marker='.', label=leg_label)
	
	plt.plot(xnew, z500, linestyle='-', color='cyan', label='M($t\\overline{t}$)=500 Est.')
	plt.plot(xnew, z350, linestyle='-', color='black', label='M($t\\overline{t}$)=350 Est.')
	
	ax.set_xticks(mthad_bin_edges)
	ax.xaxis.grid(True, which='major')
	ax.set_yticks(alpha_bin_edges)
	ax.yaxis.grid(True, which='major')
	plt.xlabel('173.1/Reco M($t_{h}$)')
	if alpha_key == 'THad_E':
		plt.ylabel('$\\alpha_{E}$ = Gen E($t_{h}$)/Reco E($t_{h}$)')
	elif alpha_key == 'THad_P':
	    plt.ylabel('$\\alpha_{P}$ = Gen P($t_{h}$)/Reco P($t_{h}$)')
	else: # 'THad_M'
	    plt.ylabel('$\\alpha_{M}$ = Gen M($t_{h}$)/Reco M($t_{h}$)')
	
	plt.ylim(0.5, 3.0)
	plt.legend(loc='upper left',fontsize=8, numpoints=1)
	fig.savefig('%s/Alpha_%s_vs_MTHad.png' % (output_dir, alpha_key) )
	#set_trace()
	#plt.show()

	ynew = np.linspace(200, 1300, 1100)
	z1p3 = f(1.3, ynew) ## estimate new alphas for mthad = 1.3
	z2p1 = f(2.1, ynew) ## estimate new alphas for mthad = 2.1
	## make alpha vs /Mttbar plots
	fig, ax = plt.subplots()
	for i in range(len(x)):
		leg_label = [ k for k,v in mthad_ranges_to_centers.items() if v == x[i] ][0]
		plt.plot(y, z[:,i], linestyle='None', color=colors[i], marker='.', label=leg_label)
	
	plt.plot(ynew, z1p3, linestyle='-', color='green', label='M($t_{h}$)=1.3 Est.')
	plt.plot(ynew, z2p1, linestyle='-', color='red', label='M($t_{h}$)=2.1 Est.')
	
	ax.set_xticks(mttbar_bin_edges)
	ax.xaxis.grid(True, which='major')
	ax.set_yticks(alpha_bin_edges)
	ax.yaxis.grid(True, which='major')
	plt.xlabel('M($t\\overline{t}$)')
	if alpha_key == 'THad_E':
		plt.ylabel('$\\alpha_{E}$ = Gen E($t_{h}$)/Reco E($t_{h}$)')
	elif alpha_key == 'THad_P':
	    plt.ylabel('$\\alpha_{P}$ = Gen P($t_{h}$)/Reco P($t_{h}$)')
	else: # 'THad_M'
	    plt.ylabel('$\\alpha_{M}$ = Gen M($t_{h}$)/Reco M($t_{h}$)')
	
	plt.ylim(0.5, 3.0)
	plt.legend(loc='upper right',fontsize=8, numpoints=1)
	fig.savefig('%s/Alpha_%s_vs_MTTBar.png' % (output_dir, alpha_key) )
	#set_trace()
