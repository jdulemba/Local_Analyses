#! /usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import os
from pdb import set_trace
import math
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--yields', action='store_true', help='Compare yields for all categories over 2016, 2017, 2018')
parser.add_argument('--diff', action='store_true', help='Compare difference in yields between lep SFs application for all categories over 2016, 2017, 2018')
parser.add_argument('--all', action='store_true', help='Perform args.yields and args.diff')

args = parser.parse_args()

## initialize global variables
jobid = os.environ['jobid']
ura_proj = os.environ['ANALYSES_PROJECT']
output_dir = os.path.join(ura_proj, 'results', jobid, 'htt')
##

leptons = ['muons', 'electrons']
njets = [('3', '3Jets'), ('4+', '4PJets')]

    ### THESE VALUES ARE HARDCODED!!! BE CAREFUL!!!
yield_fname = 'yields_and_fracs.txt'
sample_cats = ['EWK', 'single_top', 'ttJets', 'QCD', 'SIM', 'Observed', 'data/SIM']

indir_2016 = '2016Legacy/plots/htt_DeepJet_2016Legacy_j20l50MT40_1lep/preselection'
indir_2017 = '2017/plots/htt_DeepJet_2017data_j20l50MT40_1lep/preselection'
indir_2018 = '2018/plots/htt_DeepJet_2018data_j20l50MT40_1lep/preselection'
indirs = [indir_2016, indir_2017, indir_2018]

No_Reqs_2016 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}
No_Reqs_2017 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}
No_Reqs_2018 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}

No_Reqs_dicts = [No_Reqs_2016, No_Reqs_2017, No_Reqs_2018]

Before_LepSFs_2016 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}
Before_LepSFs_2017 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}
Before_LepSFs_2018 = {
    'muons' : {
        '3' : {},
        '4+' : {},
    },
    'electrons' : {
        '3' : {},
        '4+' : {},
    },
}

Before_LepSFs_dicts = [Before_LepSFs_2016, Before_LepSFs_2017, Before_LepSFs_2018]
    
for lep in leptons:
    for jmult, jdir in njets:
        for idx, indir in enumerate(indirs):
                ## info from No_Reqs dir
            no_reqs_dir = os.path.join('/Users/jdulemba/RESEARCH/htt', indir, lep, jdir, 'No_Reqs', yield_fname)
            no_reqs_yields = open(no_reqs_dir).readlines()
                ## info from Before_LepSFs dir
            before_lepsf_dir = os.path.join('/Users/jdulemba/RESEARCH/htt', indir, lep, jdir, 'Before_LepSFs', yield_fname)
            before_lepsf_yields = open(before_lepsf_dir).readlines()
            for cat in sample_cats:
                yield_ind = -1 if cat == 'data/SIM' else -2
                    ## info from No_Reqs dir
                no_reqs_line = [ i.strip() for i in no_reqs_yields if cat in i ][0].split('|') ## find line corresponding to sample category
                No_Reqs_dicts[idx][lep][jmult][cat] = float(no_reqs_line[yield_ind].strip())
                    ## info from Before_LepSFs dir
                before_lepsf_line = [ i.strip() for i in before_lepsf_yields if cat in i ][0].split('|') ## find line corresponding to sample category
                Before_LepSFs_dicts[idx][lep][jmult][cat] = float(before_lepsf_line[yield_ind].strip())

if args.yields or args.all:
    for lep in leptons:
        for jmult, jdir in njets:
            for cat in sample_cats:
                xbins = [2016, 2017, 2018]

                ## No_Reqs
                no_reqs_yields = [No_Reqs_2016[lep][jmult][cat], No_Reqs_2017[lep][jmult][cat], No_Reqs_2018[lep][jmult][cat]]
                
                    ## find interpolated value for 2017
                no_reqs_interp = interp1d([xbins[0], xbins[2]], [no_reqs_yields[0], no_reqs_yields[2]])

                    ## plot values
                fig = plt.figure()
                plt.title('%s, %sjets, %s' % (lep, jmult, cat))
                plt.ylabel('Yields')
                plt.xlabel('Year')
    
                plt.plot([xbins[0], xbins[2]], no_reqs_interp([xbins[0], xbins[2]]), label='Interpolation')
                plt.scatter(xbins, no_reqs_yields, color='k', label='Observed Values', marker='o')
    
                plt.xlim(min(xbins)-0.2, max(xbins)+0.2)
                plt.ticklabel_format(useOffset=False)
                plt.locator_params(axis='x', nbins=len(xbins))
                plt.legend(scatterpoints=1, loc='upper left', fontsize='small')
                plt.grid()
                plt.tight_layout()
                fname = 'dataOversim' if cat == 'data/SIM' else cat
                fig.savefig('%s/yields/No_Reqs/%s.png' % (output_dir, '_'.join(['run2_No_Reqs_yields_comparison', lep, jdir, fname])))
                plt.close()

                ## Before_LepSFs
                before_lepsf_yields = [Before_LepSFs_2016[lep][jmult][cat], Before_LepSFs_2017[lep][jmult][cat], Before_LepSFs_2018[lep][jmult][cat]]
                
                    ## find interpolated value for 2017
                before_lepsf_interp = interp1d([xbins[0], xbins[2]], [before_lepsf_yields[0], before_lepsf_yields[2]])

                    ## plot values
                fig = plt.figure()
                plt.title('%s, %sjets, %s' % (lep, jmult, cat))
                plt.ylabel('Yields')
                plt.xlabel('Year')
    
                plt.plot([xbins[0], xbins[2]], before_lepsf_interp([xbins[0], xbins[2]]), label='Interpolation')
                plt.scatter(xbins, before_lepsf_yields, color='k', label='Observed Values', marker='o')
    
                plt.xlim(min(xbins)-0.2, max(xbins)+0.2)
                plt.ticklabel_format(useOffset=False)
                plt.locator_params(axis='x', nbins=len(xbins))
                plt.legend(scatterpoints=1, loc='upper left', fontsize='small')
                plt.grid()
                plt.tight_layout()
                fname = 'dataOversim' if cat == 'data/SIM' else cat
                fig.savefig('%s/yields/Before_LepSFs/%s.png' % (output_dir, '_'.join(['run2_Before_LepSFs_yields_comparison', lep, jdir, fname])))
                plt.close()
                #set_trace()

if args.diff or args.all:
    for lep in leptons:
        for jmult, jdir in njets:
            for cat in sample_cats:
                #set_trace()
                xbins = [2016, 2017, 2018]
                diffs = [Before_LepSFs_2016[lep][jmult][cat]-No_Reqs_2016[lep][jmult][cat], Before_LepSFs_2017[lep][jmult][cat]-No_Reqs_2017[lep][jmult][cat], Before_LepSFs_2018[lep][jmult][cat]-No_Reqs_2018[lep][jmult][cat]]
                ratios = [Before_LepSFs_2016[lep][jmult][cat]/No_Reqs_2016[lep][jmult][cat], Before_LepSFs_2017[lep][jmult][cat]/No_Reqs_2017[lep][jmult][cat], Before_LepSFs_2018[lep][jmult][cat]/No_Reqs_2018[lep][jmult][cat]]
                
                    ## find interpolated value for 2017
                diff_interp = interp1d([xbins[0], xbins[2]], [diffs[0], diffs[2]])
                ratio_interp = interp1d([xbins[0], xbins[2]], [ratios[0], ratios[2]])
    
                    ## make plots diff yields
                fig = plt.figure()
                fig.suptitle('%s, %sjets, %s' % (lep, jmult, cat))

                ax1 = plt.subplot2grid((5,1), (0,0), rowspan=3, colspan=1)
                ax1.plot([xbins[0], xbins[2]], diff_interp([xbins[0], xbins[2]]), label='Interpolation')
                ax1.scatter(xbins, diffs, color='k', label='Observed Values', marker='o')

                ax1.set_ylabel('Yields Before Lep SF-Yields After')
                ax1.ticklabel_format(useOffset=False)
                plt.locator_params(axis='x', nbins=len(xbins))
                ax1.set_xlim(min(xbins)-0.2, max(xbins)+0.2)
                ax1.legend(scatterpoints=1, loc='upper left', fontsize='small')
                ax1.grid()

                ax2 = plt.subplot2grid((5,1), (3,0), rowspan=2, colspan=1)
                ax2.plot([xbins[0], xbins[2]], ratio_interp([xbins[0], xbins[2]]), label='Interpolation')
                ax2.scatter(xbins, ratios, color='k', label='Observed Values', marker='o')

                ax2.set_ylabel('$\\mathrm{\\frac{Yields\ Before\ Lep\ SF}{Yields\ After}}$')
                ax2.set_xlabel('Year')
                ax2.set_xlim(min(xbins)-0.2, max(xbins)+0.2)
                ax2.ticklabel_format(useOffset=False)
                plt.locator_params(axis='x', nbins=len(xbins))
                ax2.grid()
                plt.tight_layout(rect=[0, 0, 1, 0.99])

                fname = 'dataOversim' if cat == 'data/SIM' else cat
                fig.savefig('%s/diffs/%s.png' % (output_dir, '_'.join(['run2_yields_diff_comparison', lep, jdir, fname])))
                plt.close()
                #set_trace()
                
