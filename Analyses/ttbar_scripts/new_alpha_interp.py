#! /usr/bin/env python

'''
This file is meant to find median alpha values, fit them as alpha(mthad, mtt),
and dumping the resulting fit values for each bin of 172.5/mthad and mtt.
'''

import os, sys
from rootpy.io import root_open
from rootpy import asrootpy
from pdb import set_trace
from rootpy.plotting import views, Hist, Hist2D
from argparse import ArgumentParser
import ROOT
import argparse
import numpy as np
from scipy import interpolate
import Utilities.python.functions as fncts
from Utilities.python.RebinView import RebinView
import matplotlib
import matplotlib.pyplot as plt

## initialize global variables
jobid = os.environ['jobid']
ura_proj = os.environ['ANALYSES_PROJECT']
input_dir = '%s/inputs/%s/ttbar' % (ura_proj, jobid)
output_dir = '%s/results/%s/ttbar' % (ura_proj, jobid)
##

## optional input arguments
parser = argparse.ArgumentParser(description='Create plots and output root file with alpha correction')

parser.add_argument('-infile', default='ttJets', help='Choose input file to use (without .root)')
parser.add_argument('-outfile', default='alpha_hists_%s' % jobid, help='Choose output filename to use (without .root)')
args = parser.parse_args()
##

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

in_fname = '%s/%s.root' % (input_dir, args.infile)
out_fname = '%s/%s.root' % (output_dir, args.outfile) # write to results/ttbar directory

directory = '3J/nosys/Alpha_Correction/right'

## find and open ttJets file
if not os.path.isfile(in_fname):
    print '%s.root not found in %s!' % (args.infile, input_dir)
    sys.exit()

myfile = root_open(in_fname, 'read')
##


## 3D hists
fitvars = [
    ('THad_E/Alpha_THad_E_Mtt_vs_Mthad_vs_Alpha'),
    ('THad_P/Alpha_THad_P_Mtt_vs_Mthad_vs_Alpha'),
]
    ## create file to store hists for 'THad_E' and 'THad_P'
with root_open( out_fname, 'w' ) as out:# write to results/ttbar directory
    outdir = out.mkdir('nosys')
    outdir.cd()


def fit_single_mtt_medians( medians, errors, xbins, output_xbins, fit_type ):
        ## create 1D hist to extract 1d or 2d fit parameters
    if not ( fit_type == "pol1" or fit_type == "pol2"):
        print 'Only "pol1" and "pol2" are supported for fits right now!'
        sys.exit()

    medians, errors = np.array(medians), np.array(errors)
    invalid_indices = np.where( medians <= 0 )[0] ## find indices that have invalid values

    #set_trace()
    valid_medians = np.array( [ medians[ind] for ind in range(len(medians)) if ind not in invalid_indices] )
    valid_errors = np.array( [ errors[ind] for ind in range(len(errors)) if ind not in invalid_indices] )
    xbin_centers = np.array( [(xbins[i+1]+xbins[i])/2 for i in range(len(xbins)-1)] )
    fit_xbins = np.array( [ xbin_centers[ind] for ind in range(len(xbin_centers)) if ind not in invalid_indices] )

    fit_deg = 1 if fit_type=="pol1" else 2
    np_fit = np.polyfit( fit_xbins, valid_medians, fit_deg, w=np.reciprocal(valid_errors) )
    fitvals = np.poly1d(np_fit)( output_xbins )

    #set_trace()
    return fitvals



def fit_binned_mtt_medians( medians, xbins, ybins , fit_type ):
        ## create 1D hist to extract 1d or 2d fit parameters
    if not ( fit_type == "pol1" or fit_type == "pol2"):
        print 'Only "pol1" and "pol2" are supported for fits right now!'
        sys.exit()

    medians = np.array(medians)
    #invalid_indices = np.where( medians <= 0 )[0] ## find indices that have invalid values
    rows, cols = np.where( medians <= 0 )

    valid_medians = np.reshape( np.array(medians[medians > 0]), (medians.shape[0],len(np.array(medians[medians > 0]))/medians.shape[0]) )
    xbin_centers = np.array( [(xbins[i+1]+xbins[i])/2 for i in range(len(xbins)-1)] )
    fit_xbins = np.array( [xbin_centers[ind] for ind in range(len(xbin_centers)) if ind not in set(cols)] )
    ybin_centers = np.array( [(ybins[i+1]+ybins[i])/2 for i in range(len(ybins)-1)] )
    #fit_ybins = np.array( [ ybin_centers[ind] for ind in range(len(ybin_centers)) if ind not in invalid_indices] )

    #set_trace()
    fit_deg = 'linear' if fit_type=="pol1" else 'cubic'
    fit = interpolate.interp2d(fit_xbins, ybin_centers, valid_medians, kind=fit_deg)

    #set_trace()
    return fit
#set_trace()

def write_alphas_to_root(fname='', medians=None, errors=None, xbins=None, ybins=None, output_xbins=None, output_ybins=None, hname=''):

    fit_func = ""
    if '1d' in hname: fit_func = "pol1"
    elif '2d' in hname: fit_func = "pol2"
    else:
        print 'Only "pol1" and "pol2" are supported for fits right now!'
        sys.exit()

    with root_open( fname, 'update' ) as out:

        outdir = out.GetDirectory('nosys')
        outdir.cd()

        fig, ax = plt.subplots()
        fig.suptitle('$\\alpha$ Fit Values')

        alpha_hist = 0
        #set_trace() 

        ## fill alphas from fit parameters for entire mtt range
        if np.ndim(medians) == 1:
            fitvals = fit_single_mtt_medians( medians=medians, errors=errors, xbins=xbins, output_xbins=output_xbins, fit_type=fit_func )

            alpha_hist = Hist(output_xbins, name=hname, title='#alpha Fit Values')
            alpha_hist.set_x_title(hist.xaxis.title)
            alpha_hist.set_y_title(hist.zaxis.title)

            for binx in range(alpha_hist.GetXaxis().GetFirst(), alpha_hist.GetXaxis().GetLast()+1):
                alpha_hist[binx] = fitvals[binx-1]

            plt.plot(output_xbins, fitvals, color='black')
            ax.xaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='major')
            plt.xlabel('$%s$' % hist.xaxis.title)
            plt.ylabel('$%s$' % hist.zaxis.title.rstrip('#').replace('#', '\\'))
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #gets rid of title overlap with canvas
            fig.savefig('%s/%s_AlphaFit.png' % (output_dir, hname) )
            #set_trace()

        ## fill alphas from fit parameters for single mtt bin
        elif np.ndim(medians) == 2:
            interp_fit = fit_binned_mtt_medians( medians=medians, xbins=xbins, ybins=ybins, fit_type=fit_func )

            alpha_hist = Hist2D(output_xbins, output_ybins, name=hname, title='#alpha Fit Values')
            alpha_hist.set_x_title(hist.xaxis.title)
            alpha_hist.set_y_title(hist.yaxis.title)

	        #set_trace()
            for biny in range( alpha_hist.GetYaxis().GetFirst(), alpha_hist.GetYaxis().GetLast()+1 ):
                for binx in range(alpha_hist.GetXaxis().GetFirst(), alpha_hist.GetXaxis().GetLast()+1):
                    alpha_hist[binx, biny] = interp_fit(output_xbins[binx-1], output_ybins[biny-1])


            xnew, ynew = np.mgrid[output_xbins.min():output_xbins.max():500j, output_ybins.min():output_ybins.max():500j]
            plt.pcolor( xnew, ynew, interp_fit(output_xbins, output_ybins).T )
            cbar = plt.colorbar()
            cbar.set_label('$%s$' % hist.zaxis.title.rstrip('#').replace('#', '\\').split('=')[0], rotation=0)
            ax.xaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='major')
            plt.xlabel('$%s$' % hist.xaxis.title)
            plt.ylabel('$%s$' % hist.yaxis.title.rstrip('#').replace('#', '\\'))
            fig.savefig('%s/%s_AlphaFit.png' % (output_dir, hname) )
            #set_trace()

        else:
            print 'Something bad happened.'
            sys.exit()

        outdir.WriteTObject(alpha_hist, hname)
        #alpha_hist.Write()
        print '\n%s written to %s' % (hname, fname)

    out.Close()


#set_trace()

for hvar in fitvars:
    hname = '/'.join([directory, hvar])
    hist = asrootpy(myfile.Get(hname)).Clone()
    
    if hist.Integral() == 0:
        continue
    
        ## define bin edges for future rebinning
    mthad_bins = np.linspace( hist.GetXaxis().GetBinLowEdge(1), hist.GetXaxis().GetBinUpEdge(hist.GetNbinsX()), hist.GetNbinsX()+1 )
    #mtt_bins = np.linspace( hist.GetYaxis().GetBinLowEdge(1), hist.GetYaxis().GetBinUpEdge(hist.GetNbinsY()), hist.GetNbinsY()+1 )
    alpha_bins = np.linspace( hist.GetZaxis().GetBinLowEdge(1), hist.GetZaxis().GetBinUpEdge(hist.GetNbinsZ()), hist.GetNbinsZ()+1 )
    
    #set_trace()
    
    #mthad_bins = np.linspace(0.9, 2.5, 9)
    mtt_bins = np.array([200., 350., 400., 500., 700., 1000., 2000.])
    hist = RebinView.newRebin3D(hist, mthad_bins, mtt_bins, alpha_bins)
    
    mthad_out_bins = np.linspace(min(mthad_bins), max(mthad_bins), 500)
    mtt_out_bins = np.linspace(min(mtt_bins), max(mtt_bins), 500)
    
        ## get medians for single bin of mtt
    medians, median_errors = fncts.median_from_3d_hist(hist, projection='zx', xbins=mthad_bins, ybins=alpha_bins)
    
    
        ## get medians for all bins of mtt
    binned_mtt_medians = np.zeros( ( hist.GetNbinsY(), hist.GetNbinsX() ) )
    binned_mtt_errors = np.zeros( ( hist.GetNbinsY(), hist.GetNbinsX() ) )
    for ybin in range(1, hist.GetNbinsY()+1):
    
        h3d_yslice = hist.Clone()
        h3d_yslice.GetYaxis().SetRange(ybin, ybin+1)
        
        meds, med_errors = fncts.median_from_3d_hist(h3d_yslice, projection='zx', xbins=mthad_bins, ybins=alpha_bins)
        binned_mtt_medians[ybin-1] = meds
        binned_mtt_errors[ybin-1] = med_errors
        
        ## make hists and write them to a root file
    for fit_degree in ['1d', '2d']:
            ## entire mtt range
        write_alphas_to_root(fname=out_fname, medians=medians, errors=median_errors, xbins=mthad_bins, output_xbins=mthad_out_bins, hname='%s_All_%s' % (hvar.split('/')[0], fit_degree) )
            ## mtt range binned
        write_alphas_to_root(fname=out_fname, medians=binned_mtt_medians, errors=binned_mtt_errors, xbins=mthad_bins, ybins=mtt_bins, output_xbins=mthad_out_bins, output_ybins=mtt_out_bins, hname='%s_Mtt_%s' % (hvar.split('/')[0], fit_degree) )
    
#set_trace()
