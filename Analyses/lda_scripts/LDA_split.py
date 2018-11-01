from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from scipy import linalg
import root_numpy as rtnp
import numpy as np
import os, glob, sys
from rootpy import asrootpy
from rootpy.io import root_open
from pdb import set_trace
import argparse
import logging
import print_table as pt
import prettyjson as prettyjson

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='Choose the ttJets file to use for the analysis (ttJetsM0, ttJetM700, ttJetsM1000).')
parser.add_argument('--disc_type', help='Choose to compare THadPt with TotalDisc, NSDisc, or MassDisc (Total, NS, or Mass).')
parser.add_argument('--comp_type', help='Choose to compare:\n   Same discrimnant treatment across different event types (event)\n       Same event type but different discriminant treatment (treatment)')
args = parser.parse_args()


if not (args.disc_type == 'Total' or args.disc_type == 'Mass' or args.disc_type == 'NS'):
        logging.error('Not a valid discriminant type to choose from.')
        sys.exit()

if not (args.comp_type == 'event' or args.comp_type == 'treatment'):
        logging.error('Not a valid event comparison type to choose from.')
        sys.exit()

ytitle = ''
discname = ''
if args.disc_type == 'Total':
        ytitle = '$\lambda_{comb}$'
        discname = 'Totaldisc'
if args.disc_type == 'NS':
        ytitle = '$\lambda_{NS}$'
        discname = 'NSdisc'
if args.disc_type == 'Mass':
        ytitle = '$\lambda_{M}$'
        discname = 'Massdisc'

#filename = '~/ROOT/ttbar_reco_3J/ttJetsM1000.ttbar_reco_3J.test.root'
filename = '~/ROOT/ttbar_reco_3J/%s.ttbar_matched_perms.test.root' % args.file

#rows = []
json = {
    'comp_type' : args.comp_type,
    'disc_type' : args.disc_type,
}

################################################################################
# create arrays for valid variable distributions and classification types
def var_dists(cat1_varlist, cat2_varlist):
        THadPt_0 = rtnp.root2array(filename, treename=cat1_varlist[0], branches=cat1_varlist[1])
        THadPt_1 = rtnp.root2array(filename, treename=cat2_varlist[0], branches=cat2_varlist[1])
        Disc_0 = rtnp.root2array(filename, treename=cat1_varlist[0], branches=cat1_varlist[2])
        Disc_1 = rtnp.root2array(filename, treename=cat2_varlist[0], branches=cat2_varlist[2])

                ## find variable distributions where solution exists
        valid_Disc0_indices = [i for i, j in enumerate(Disc_0) if j < 1e100]
        valid_Disc1_indices = [i for i, j in enumerate(Disc_1) if j < 1e100]

        valid_Disc_0 = Disc_0[valid_Disc0_indices]
        valid_THadPt_0 = THadPt_0[valid_Disc0_indices]

        valid_Disc_1 = Disc_1[valid_Disc1_indices]
        valid_THadPt_1 = THadPt_1[valid_Disc1_indices]

        Disc0_type = np.zeros(len(valid_Disc_0))
        Disc1_type = np.ones(len(valid_Disc_1))

        Discs = np.concatenate((valid_Disc_0, valid_Disc_1), axis=0)
        THadPts = np.concatenate((valid_THadPt_0, valid_THadPt_1), axis=0)

        vars_dist = np.vstack((THadPts, Discs)).T
        class_type = np.concatenate((Disc0_type, Disc1_type), axis=0)

                ## find variable distributions including events without solution
        All_Disc0_type = np.zeros(len(Disc_0))
        All_Disc1_type = np.ones(len(Disc_1))

        All_Discs = np.concatenate((Disc_0, Disc_1), axis=0)
        All_THadPts = np.concatenate((THadPt_0, THadPt_1), axis=0)

        All_vars_dist = np.vstack((All_THadPts, All_Discs)).T
        All_class_type = np.concatenate((All_Disc0_type, All_Disc1_type), axis=0)

        invalid_Disc0_indices = [i for i, j in enumerate(Disc_0) if j >= 1e100]
        invalid_Disc1_indices = [i for i, j in enumerate(Disc_1) if j >= 1e100]

        #set_trace()

        return vars_dist, class_type, All_vars_dist, All_class_type, valid_Disc0_indices, valid_Disc1_indices, invalid_Disc0_indices, invalid_Disc1_indices



##################################################################################


## create dicts for all 4 categories
Merged_TreatMerged = {'treename' : 'Merged_TreatMerged', 'THad_Pt' : 'M_TM_THad_Pt', 'Discs' : ['M_TM_Totaldisc', 'M_TM_Massdisc', 'M_TM_NSdisc'], 'Comp_types' : ['TreatMerged', 'Merged']}
Merged_TreatLost = {'treename' : 'Merged_TreatLost', 'THad_Pt' : 'M_TL_THad_Pt', 'Discs' : ['M_TL_Totaldisc', 'M_TL_Massdisc', 'M_TL_NSdisc'], 'Comp_types' : ['TreatLost', 'Merged']}
Lost_TreatMerged = {'treename' : 'Lost_TreatMerged', 'THad_Pt' : 'L_TM_THad_Pt', 'Discs' : ['L_TM_Totaldisc', 'L_TM_Massdisc', 'L_TM_NSdisc'], 'Comp_types' : ['TreatMerged', 'Lost']}
Lost_TreatLost = {'treename' : 'Lost_TreatLost', 'THad_Pt' : 'L_TL_THad_Pt', 'Discs' : ['L_TL_Totaldisc', 'L_TL_Massdisc', 'L_TL_NSdisc'], 'Comp_types' : ['TreatLost', 'Lost']}

## get discriminant type to use for 2D comparison
disctype = [i for i in Merged_TreatMerged.get('Discs') if args.disc_type in i]
disc_idx = Merged_TreatMerged.get('Discs').index(disctype[0])

## find comparison type for hist titles/labels
comptype = []

if args.comp_type == 'event':
        comptype = [i for i in Merged_TreatMerged.get('Comp_types') if 'Treat' in i]
if args.comp_type == 'treatment':
        comptype = [i for i in Merged_TreatMerged.get('Comp_types') if 'Treat' not in i]

comp_idx = Merged_TreatMerged.get('Comp_types').index(comptype[0])

#set_trace()
## creat lists of variables for all 4 categories (treename, THadPt branchname, discr branchname, plot title, plot label)
M_TM_list = [Merged_TreatMerged['treename'], Merged_TreatMerged['THad_Pt'], Merged_TreatMerged['Discs'][disc_idx], Merged_TreatMerged['Comp_types'][comp_idx], Merged_TreatMerged['Comp_types'][~comp_idx]]
L_TM_list = [Lost_TreatMerged['treename'], Lost_TreatMerged['THad_Pt'], Lost_TreatMerged['Discs'][disc_idx], Lost_TreatMerged['Comp_types'][comp_idx], Lost_TreatMerged['Comp_types'][~comp_idx]]
M_TL_list = [Merged_TreatLost['treename'], Merged_TreatLost['THad_Pt'], Merged_TreatLost['Discs'][disc_idx], Merged_TreatLost['Comp_types'][comp_idx], Merged_TreatLost['Comp_types'][~comp_idx]]
L_TL_list = [Lost_TreatLost['treename'], Lost_TreatLost['THad_Pt'], Lost_TreatLost['Discs'][disc_idx], Lost_TreatLost['Comp_types'][comp_idx], Lost_TreatLost['Comp_types'][~comp_idx]]


## create list containing list of variables whose order depends on which type of comparison is being made
cat_type_list = []
if args.comp_type == 'event':
        cat_type_list = [[M_TM_list, L_TM_list], [M_TL_list, L_TL_list]]

if args.comp_type == 'treatment':
        cat_type_list = [[M_TM_list, M_TL_list], [L_TM_list, L_TL_list]]

outname = '%s_THadPt_vs_%s_%s_%s_Comparison_split' % (args.file, discname, cat_type_list[0][0][3], cat_type_list[1][0][3])
out_dir = '/home/jdulemba/ROOT/ttbar_reco_3J/plots/Split'

for i in range(len(cat_type_list)):
        vars_dist, class_type, All_vars_dist, All_class_type, valid_disc0_indices, valid_disc1_indices, invalid_disc0_indices, invalid_disc1_indices = var_dists(cat_type_list[i][0], cat_type_list[i][1])
        #vars_dist, class_type = var_dists(cat_type_list[i][0], cat_type_list[i][1])
	#X_train, X_test, y_train, y_test = train_test_split(vars_dist[:,0], vars_dist[:,1], test_size = 0.2)
	#X_train, X_test, y_train, y_test = train_test_split(vars_dist, class_type, test_size = 0.2)

        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                      solver='svd', store_covariance=True, tol=0.0001)
        #class_type_pred = lda.fit(X_train, y_train).predict(X_test)
	##lin_score = lda.score(vars_dist[train_index,:], class_type[train_index])
	#lin_score = lda.score(X_test, y_test)
	#print lin_score


	kf = KFold(n_splits = 10, random_state=None, shuffle=False)
	figure = plt.figure()
	lin_scores=[]
	for train_index, test_index in kf.split(vars_dist[:,0]):
        	class_type_pred = lda.fit(vars_dist[train_index,:], class_type[train_index]).predict(vars_dist[test_index,:])
		#lin_score = lda.score(vars_dist[train_index,:], class_type[train_index])
		lin_score = lda.score(vars_dist[test_index,:], class_type[test_index])
		#print lin_score
		lin_scores.append(lin_score)
	#n, bins, patches = plt.hist([X_train,X_test], 50, histtype='bar', stacked=True)#, facecolor='green')
	#n, bins, patches = plt.hist(vars_dist[:,0], 50, facecolor='green')
	n, bins, patches = plt.hist(lin_scores, 5, facecolor='green')
	print '%s Results mean = %f, stdev = %f' % (cat_type_list[i][0][3], np.mean(lin_scores), np.std(lin_scores))

#	with open('%s/%s.json' % (out_dir, outname), 'w') as f:
#		set_trace()
#
#	plt.show()
#	set_trace()




