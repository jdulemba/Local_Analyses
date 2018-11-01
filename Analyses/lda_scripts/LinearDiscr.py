from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
parser.add_argument('--comp_type', help='Choose to compare:\n	Same discrimnant treatment across different event types (event)\n	Same event type but different discriminant treatment (treatment)')
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
# function that calculates the fractions of events in the matrix T_M, T_L, F_M, F_L

### events in which only one solution exists
def single_solution( Merged_Events, Lost_Events ):
	nevents = len(Merged_Events[0,:])+len(Lost_Events[0,:])

	MergedEvents_TrueMerged_TrueLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_TrueMerged_FalseLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_FalseMerged_TrueLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_FalseMerged_FalseLost = np.zeros(len(Merged_Events[0,:]))

	## events with first index==3 are T_M, second index==3 are T_L
	for i in range(len(Merged_Events[0,:])):
		MergedEvents_TrueMerged_TrueLost[i] = np.array_equal(Merged_Events[:,i], np.array([3,0])) or np.array_equal(Merged_Events[:,i], np.array([1,3]))
		MergedEvents_TrueMerged_FalseLost[i] = np.array_equal(Merged_Events[:,i], np.array([3,1]))
		MergedEvents_FalseMerged_TrueLost[i] = np.array_equal(Merged_Events[:,i], np.array([0,0])) or np.array_equal(Merged_Events[:,i], np.array([0,3])) #[0,0] shouldn't occur
		MergedEvents_FalseMerged_FalseLost[i] = np.array_equal(Merged_Events[:,i], np.array([0,1])) # shouldn't occur

	LostEvents_TrueMerged_TrueLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_TrueMerged_FalseLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_FalseMerged_TrueLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_FalseMerged_FalseLost = np.zeros(len(Lost_Events[0,:]))
	
	for i in range(len(Lost_Events[0,:])):
		LostEvents_TrueMerged_TrueLost[i] = np.array_equal(Lost_Events[:,i], np.array([3,0])) or np.array_equal(Lost_Events[:,i], np.array([1,3]))
		LostEvents_TrueMerged_FalseLost[i] = np.array_equal(Lost_Events[:,i], np.array([3,1]))
		LostEvents_FalseMerged_TrueLost[i] = np.array_equal(Lost_Events[:,i], np.array([0,0])) or np.array_equal(Lost_Events[:,i], np.array([0,3])) #[0,0] shouldn't occur
		LostEvents_FalseMerged_FalseLost[i] = np.array_equal(Lost_Events[:,i], np.array([0,1])) #shouldn't occur

	TrueMerged_TrueLost = (MergedEvents_TrueMerged_TrueLost == 1).sum() + (LostEvents_TrueMerged_TrueLost == 1).sum()
	TrueMerged_FalseLost = (MergedEvents_TrueMerged_FalseLost == 1).sum() + (LostEvents_TrueMerged_FalseLost == 1).sum()
	FalseMerged_TrueLost = (MergedEvents_FalseMerged_TrueLost == 1).sum() + (LostEvents_FalseMerged_TrueLost == 1).sum()
	FalseMerged_FalseLost = (MergedEvents_FalseMerged_FalseLost == 1).sum() + (LostEvents_FalseMerged_FalseLost == 1).sum()

	TrueMerged_TrueLost_Frac = float(TrueMerged_TrueLost)/nevents
	TrueMerged_FalseLost_Frac = float(TrueMerged_FalseLost)/nevents
	FalseMerged_TrueLost_Frac = float(FalseMerged_TrueLost)/nevents
	FalseMerged_FalseLost_Frac = float(FalseMerged_FalseLost)/nevents

	json["Single Solution Event Classifications"] = {}
	json["Single Solution Event Classifications"]["TrueMerged_TrueLost"] = TrueMerged_TrueLost
	json["Single Solution Event Classifications"]["TrueMerged_FalseLost"] = TrueMerged_FalseLost
	json["Single Solution Event Classifications"]["FalseMerged_TrueLost"] = FalseMerged_TrueLost
	json["Single Solution Event Classifications"]["FalseMerged_FalseLost"] = FalseMerged_FalseLost
	json["Single Solution Event Classifications"]["TrueMerged_TrueLost_Frac"] = TrueMerged_TrueLost_Frac
	json["Single Solution Event Classifications"]["TrueMerged_FalseLost_Frac"] = TrueMerged_FalseLost_Frac
	json["Single Solution Event Classifications"]["FalseMerged_TrueLost_Frac"] = FalseMerged_TrueLost_Frac
	json["Single Solution Event Classifications"]["FalseMerged_FalseLost_Frac"] = FalseMerged_FalseLost_Frac

	return TrueMerged_TrueLost, TrueMerged_FalseLost, FalseMerged_TrueLost, FalseMerged_FalseLost, nevents


#### events in which both solutions exist
def two_solutions( Merged_Events, Lost_Events ):
	nevents = len(Merged_Events[0,:])+len(Lost_Events[0,:])

	MergedEvents_TrueMerged_TrueLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_TrueMerged_FalseLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_FalseMerged_TrueLost = np.zeros(len(Merged_Events[0,:]))
	MergedEvents_FalseMerged_FalseLost = np.zeros(len(Merged_Events[0,:]))
	
	for i in range(len(Merged_Events[0,:])):
		MergedEvents_TrueMerged_TrueLost[i] = np.array_equal(Merged_Events[:,i], np.array([1,0]))
		MergedEvents_TrueMerged_FalseLost[i] = np.array_equal(Merged_Events[:,i], np.array([1,1]))
		MergedEvents_FalseMerged_TrueLost[i] = np.array_equal(Merged_Events[:,i], np.array([0,0]))
		MergedEvents_FalseMerged_FalseLost[i] = np.array_equal(Merged_Events[:,i], np.array([0,1]))

	LostEvents_TrueMerged_TrueLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_TrueMerged_FalseLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_FalseMerged_TrueLost = np.zeros(len(Lost_Events[0,:]))
	LostEvents_FalseMerged_FalseLost = np.zeros(len(Lost_Events[0,:]))
	
	for i in range(len(Lost_Events[0,:])):
		LostEvents_TrueMerged_TrueLost[i] = np.array_equal(Lost_Events[:,i], np.array([1,0]))
		LostEvents_TrueMerged_FalseLost[i] = np.array_equal(Lost_Events[:,i], np.array([1,1]))
		LostEvents_FalseMerged_TrueLost[i] = np.array_equal(Lost_Events[:,i], np.array([0,0]))
		LostEvents_FalseMerged_FalseLost[i] = np.array_equal(Lost_Events[:,i], np.array([0,1]))

	TrueMerged_TrueLost = (MergedEvents_TrueMerged_TrueLost == 1).sum() + (LostEvents_TrueMerged_TrueLost == 1).sum()
	TrueMerged_FalseLost = (MergedEvents_TrueMerged_FalseLost == 1).sum() + (LostEvents_TrueMerged_FalseLost == 1).sum()
	FalseMerged_TrueLost = (MergedEvents_FalseMerged_TrueLost == 1).sum() + (LostEvents_FalseMerged_TrueLost == 1).sum()
	FalseMerged_FalseLost = (MergedEvents_FalseMerged_FalseLost == 1).sum() + (LostEvents_FalseMerged_FalseLost == 1).sum()

	TrueMerged_TrueLost_Frac = float(TrueMerged_TrueLost)/nevents
	TrueMerged_FalseLost_Frac = float(TrueMerged_FalseLost)/nevents
	FalseMerged_TrueLost_Frac = float(FalseMerged_TrueLost)/nevents
	FalseMerged_FalseLost_Frac = float(FalseMerged_FalseLost)/nevents

	json["Two Solution Event Classifications"] = {}
	json["Two Solution Event Classifications"]["TrueMerged_TrueLost"] = TrueMerged_TrueLost
	json["Two Solution Event Classifications"]["TrueMerged_FalseLost"] = TrueMerged_FalseLost
	json["Two Solution Event Classifications"]["FalseMerged_TrueLost"] = FalseMerged_TrueLost
	json["Two Solution Event Classifications"]["FalseMerged_FalseLost"] = FalseMerged_FalseLost
	json["Two Solution Event Classifications"]["TrueMerged_TrueLost_Frac"] = TrueMerged_TrueLost_Frac
	json["Two Solution Event Classifications"]["TrueMerged_FalseLost_Frac"] = TrueMerged_FalseLost_Frac
	json["Two Solution Event Classifications"]["FalseMerged_TrueLost_Frac"] = FalseMerged_TrueLost_Frac
	json["Two Solution Event Classifications"]["FalseMerged_FalseLost_Frac"] = FalseMerged_FalseLost_Frac

	return TrueMerged_TrueLost, TrueMerged_FalseLost, FalseMerged_TrueLost, FalseMerged_FalseLost, nevents
	

def category_classifications( Merged_Events, Lost_Events ):
	#set_trace()
	#nevents = len(Merged_Events[0,:])+len(Lost_Events[0,:])

	merged_events_one_solution = []
	for i in range(Merged_Events.ndim):
		merged_events_one_solution_indices = [x for x in range(len(Merged_Events[i,:])) if Merged_Events[i,x] == 3]
		merged_events_one_solution.append(merged_events_one_solution_indices)
	merged_events_one_solution = list(set(merged_events_one_solution[0]).union(set(merged_events_one_solution[1])))
	merged_events_two_solutions = [ x for x in range(len(Merged_Events[0,:])) if x not in merged_events_one_solution ]

	lost_events_one_solution = []
	for i in range(Lost_Events.ndim):
		lost_events_one_solution_indices = [x for x in range(len(Lost_Events[i,:])) if Lost_Events[i,x] == 3]
		lost_events_one_solution.append(lost_events_one_solution_indices)
	lost_events_one_solution = list(set(lost_events_one_solution[0]).union(set(lost_events_one_solution[1])))
	lost_events_two_solutions = [ x for x in range(len(Lost_Events[0,:])) if x not in lost_events_one_solution ]

	single_TM_TL, single_TM_FL, single_FM_TL, single_FM_FL, single_nevents = single_solution( Merged_Events[:, merged_events_one_solution], Lost_Events[:, lost_events_one_solution] )
	two_TM_TL, two_TM_FL, two_FM_TL, two_FM_FL, two_nevents = two_solutions( Merged_Events[:, merged_events_two_solutions], Lost_Events[:, lost_events_two_solutions] )
	
	#set_trace()

	nevents = single_nevents + two_nevents

	TrueMerged_TrueLost = single_TM_TL + two_TM_TL
	TrueMerged_FalseLost = single_TM_FL + two_TM_FL
	FalseMerged_TrueLost = single_FM_TL + two_FM_TL
	FalseMerged_FalseLost = single_FM_FL + two_FM_FL

	TrueMerged_TrueLost_Frac = float(TrueMerged_TrueLost)/nevents
	TrueMerged_FalseLost_Frac = float(TrueMerged_FalseLost)/nevents
	FalseMerged_TrueLost_Frac = float(FalseMerged_TrueLost)/nevents
	FalseMerged_FalseLost_Frac = float(FalseMerged_FalseLost)/nevents

	json["Combined Event Classifications"] = {}
	json["Combined Event Classifications"]["TrueMerged_TrueLost"] = TrueMerged_TrueLost
	json["Combined Event Classifications"]["TrueMerged_FalseLost"] = TrueMerged_FalseLost
	json["Combined Event Classifications"]["FalseMerged_TrueLost"] = FalseMerged_TrueLost
	json["Combined Event Classifications"]["FalseMerged_FalseLost"] = FalseMerged_FalseLost
	json["Combined Event Classifications"]["TrueMerged_TrueLost_Frac"] = TrueMerged_TrueLost_Frac
	json["Combined Event Classifications"]["TrueMerged_FalseLost_Frac"] = TrueMerged_FalseLost_Frac
	json["Combined Event Classifications"]["FalseMerged_TrueLost_Frac"] = FalseMerged_TrueLost_Frac
	json["Combined Event Classifications"]["FalseMerged_FalseLost_Frac"] = FalseMerged_FalseLost_Frac

###############################################################################
# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


###############################################################################
# plot functions
def plot_data(lda, distribution, true_dist_type, dist_type_pred, plot_title, class0_label, class1_label, fig_index):

    splot = plt.subplot(2,1,fig_index)

    plt.title('Linear %s Comp' % plot_title)
    #print(plot_title)

#    if fig_index == 1:
#        plt.title('Linear %s Comp' % plot_title)
##       plt.xlabel('$m_{b_h}$ [GeV]')
##        plt.ylabel('$m_{b_h+jet}$ [GeV]')
#        print('Linear')
#
#    elif fig_index == 2:
#        plt.title('Quadratic')
##       plt.xlabel('$m_{b_h}$ [GeV]')
##        plt.ylabel('$m_{b_h+jet}$ [GeV]')
#        print('Quadratic')

    tp = (true_dist_type == dist_type_pred)  # True Positive
    tp0, tp1 = tp[true_dist_type == 0], tp[true_dist_type == 1]  ## true positives for both categories
    X0, X1 = distribution[true_dist_type == 0], distribution[true_dist_type == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0] #merged true, false positive
    X1_tp, X1_fp = X1[tp1], X1[~tp1] #lost true, false positive
    correct_class0_eff = float(np.size(X0_tp)/2)/(float(np.size(X0_tp)/2+np.size(X0_fp)/2))
    correct_class1_eff = float(np.size(X1_tp)/2)/(float(np.size(X1_tp)/2+np.size(X1_fp)/2))

    #print("    %s True: {}".format(int(np.size(X0_tp))/int(2)) % class0_label)
    #print("    %s False: {}".format(int(np.size(X0_fp))/int(2)) % class0_label)
    #print("    %s True: {}".format(int(np.size(X1_tp))/int(2)) % class1_label)
    #print("    %s False: {}".format(int(np.size(X1_fp))/int(2)) % class1_label)
    #print("    Correct %s Efficiency = %f" % (class0_label, correct_class0_eff) )
    #print("    Correct %s Efficiency = %f" % (class1_label, correct_class1_eff) )
    #print("    Linear Boundary equation: y = %f*x + %f" % (-float(lda.coef_[0][0])/float(lda.coef_[0][1]), -float(lda.intercept_)/float(lda.coef_[0][1])))
    #print('    LDA score = %f' % lin_score)

#    rows.append((plot_title, "%s True: {}".format(int(np.size(X0_tp))/int(2)) % class0_label, "%s False: {}".format(int(np.size(X0_fp))/int(2)) % class0_label,\
#		"Correct %s Efficiency = %f" % (class0_label, correct_class0_eff),\
#		"%s True: {}".format(int(np.size(X1_tp))/int(2)) % class1_label, "%s False: {}".format(int(np.size(X1_fp))/int(2)) % class1_label,\
#		"Correct %s Efficiency = %f" % (class1_label, correct_class1_eff),\
#		"Linear Boundary equation: y = %f*x + %f" % (-float(lda.coef_[0][0])/float(lda.coef_[0][1]), -float(lda.intercept_)/float(lda.coef_[0][1])),\
#		"LDA score = %f" % lin_score))

    json[plot_title] = {}
    json[plot_title]['%s True' % class0_label] = int(np.size(X0_tp))/int(2)
    json[plot_title]['%s False' % class0_label] = int(np.size(X0_fp))/int(2)
    json[plot_title]['Correct %s Efficiency' % class0_label] = correct_class0_eff
    json[plot_title]['%s True' % class1_label] = int(np.size(X1_tp))/int(2)
    json[plot_title]['%s False' % class1_label] = int(np.size(X1_fp))/int(2)
    json[plot_title]['Correct %s Efficiency' % class1_label] = correct_class1_eff
    json[plot_title]['LDA equation'] = 'y = %f*x + %f' % (-float(lda.coef_[0][0])/float(lda.coef_[0][1]), -float(lda.intercept_)/float(lda.coef_[0][1]))
    json[plot_title]['LDA score'] = lin_score

    alpha = 0.5 #transparency of points for plotting

    # class 0 : dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], label='%s True Positive' % class0_label, linestyle='None', marker='o', alpha=alpha,
             color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], label='%s False Pos' % class0_label, linestyle='None', marker='*', alpha=alpha,
             color='#990000')  # dark red

    # class 1 : dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], label='%s True Pos' % class1_label, linestyle='None', marker='o', alpha=alpha,
             color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], label='%s False Pos' % class1_label, linestyle='None', marker='*', alpha=alpha,
             color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    plt.legend(loc='upper right',fontsize=8, numpoints=1)
    #set_trace()
    return splot, tp0, tp1


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color, edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    #splot.set_xticks([0,50,100,150,200,250])
    #splot.set_yticks([0,50,100,150,200,250,300])


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')


###############################################################################

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


vars_dist, class_type, All_vars_dist, All_class_type, valid_disc0_indices, valid_disc1_indices, invalid_disc0_indices, invalid_disc1_indices = var_dists(cat_type_list[0][0], cat_type_list[0][1])
All_tp0_category_classifications = 3*np.ones( (2, ( All_class_type == 0 ).sum()) )
All_tp1_category_classifications = 3*np.ones( (2, ( All_class_type == 1 ).sum()) )
#vars_dist, class_type = var_dists(cat_type_list[0][0], cat_type_list[0][1])
#tp0_category_classifications = np.zeros( (2, ( class_type == 0 ).sum()) )
#tp1_category_classifications = np.zeros( (2, ( class_type == 1 ).sum()) )

#set_trace()
fig = plt.figure()
for i in range(len(cat_type_list)):
	vars_dist, class_type, All_vars_dist, All_class_type, valid_disc0_indices, valid_disc1_indices, invalid_disc0_indices, invalid_disc1_indices = var_dists(cat_type_list[i][0], cat_type_list[i][1])
	#vars_dist, class_type = var_dists(cat_type_list[i][0], cat_type_list[i][1])

	# Linear Discriminant Analysis
	lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
	              solver='svd', store_covariance=True, tol=0.0001)
	class_type_pred = lda.fit(vars_dist, class_type).predict(vars_dist)
	
	lin_score = lda.score(vars_dist, class_type)
	
	#set_trace()
	#splot = plot_data(lda, vars_dist, class_type, class_type_pred, cat_type_list[i][0][3], cat_type_list[i][0][4], cat_type_list[i][1][4], fig_index=i+1)
	#splot, tp0_category_classifications[i,:], tp1_category_classifications[i,:]  = plot_data(lda, vars_dist, class_type, class_type_pred, cat_type_list[i][0][3], cat_type_list[i][0][4], cat_type_list[i][1][4], fig_index=i+1)
	splot, All_tp0_category_classifications[i,valid_disc0_indices], All_tp1_category_classifications[i,valid_disc1_indices]  = plot_data(lda, vars_dist, class_type, class_type_pred, cat_type_list[i][0][3], cat_type_list[i][0][4], cat_type_list[i][1][4], fig_index=i+1)
	#set_trace()

	plot_lda_cov(lda, splot)
	#plt.axis('tight')
	#plt.axis([0, 1000, -10, 30])
	
	plt.ylabel(ytitle)
	#plt.xlabel('$t_{h}$ $p_{T}$ [GeV]')

plt.xlabel('$t_{h}$ $p_{T}$ [GeV]')
plt.tight_layout()
#plt.show()

#category_classifications(tp0_category_classifications, tp1_category_classifications)
category_classifications(All_tp0_category_classifications, All_tp1_category_classifications)

#print 'ttJetsM1000_THadPt_vs_%s_%s_%s_Comparison.png' % (discname, cat_type_list[0][0][3], cat_type_list[1][0][3])
#set_trace()
outname = '%s_THadPt_vs_%s_%s_%s_Comparison' % (args.file, discname, cat_type_list[0][0][3], cat_type_list[1][0][3])
out_dir = '/home/jdulemba/ROOT/ttbar_reco_3J/plots'
fig.savefig('%s/%s.png' % (out_dir, outname))
#pt.print_table(rows, filename='%s.txt' % outname)

with open('%s/%s.json' % (out_dir, outname), 'w') as f:
	f.write(prettyjson.dumps(json))

print '   %s/%s.png and\n   %s/%s.json created.' % (out_dir, outname, out_dir, outname)
#set_trace()
