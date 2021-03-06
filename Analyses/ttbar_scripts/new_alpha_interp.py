'''
This file is meant to find median alpha values, fit them as alpha(mthad, mtt),
and dumping the resulting fit values for each bin of 173.1/mthad and mtt.
'''
import os, glob, sys, inspect
from rootpy.io import root_open
from rootpy import asrootpy
from pdb import set_trace
from rootpy.plotting import views, Hist, Hist2D
from argparse import ArgumentParser
import ROOT
import argparse
import numpy as np
from scipy import interpolate

# import modules from parent dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import functions as fncts
from PlotTools_python_views_RebinView import RebinView


## initialize global variables
jobid = os.getcwd()
outdir = '%s/Plots' % jobid
#project = os.environ['URA_PROJECT']
#analyzer = 'ttbar_alpha_reco'
directory = '/'.join(['3J', 'nosys', 'Alpha_Correction', 'CORRECT_WJET_CORRECT_Bs'])
##

## find and open ttJets file
results_files = filter(lambda x: 'ttJets.root' in x, os.listdir(jobid))
if not results_files:
    print "ttJets.root not found in %s.\n" % jobid
results_file  = '%s/%s' % (jobid, results_files[0])
myfile = root_open(results_file, 'read')
##

out_fname = '%s/alpha_hists' % outdir # write to $jobid/INPUT directory

## 3D hists
fitvars = [
    ('THad_E/Alpha_THad_E_Mtt_vs_Mthad_vs_Alpha'),
    ('THad_P/Alpha_THad_P_Mtt_vs_Mthad_vs_Alpha'),
    ('THad_M/Alpha_THad_M_Mtt_vs_Mthad_vs_Alpha'),
]
    ## create file to store hists for 'THad_E', 'THad_P', and 'THad_M'
with root_open('%s.root' % out_fname, 'w') as out:# write to $jobid/INPUT directory
    out.cd()

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
    #fit_deg = 1 if fit_type=="pol1" else 2
    #fit = interpolate.interpolate.RectBivariateSpline(fit_xbins, fit_ybins, np.transpose(valid_medians), s=fit_deg )

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

    with root_open('%s.root' % fname, 'update') as out:

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

	    #set_trace()

            ## fill alphas from fit parameters for single mtt bin
        elif np.ndim(medians) == 2:
	    interp_fit = fit_binned_mtt_medians( medians=medians, xbins=xbins, ybins=ybins, fit_type=fit_func )
	    #set_trace()

            alpha_hist = Hist2D(output_xbins, output_ybins, name=hname, title='#alpha Fit Values')
            alpha_hist.set_x_title(hist.xaxis.title)
            alpha_hist.set_y_title(hist.yaxis.title)

	    #set_trace()
            for biny in range( alpha_hist.GetYaxis().GetFirst(), alpha_hist.GetYaxis().GetLast()+1 ):
                for binx in range(alpha_hist.GetXaxis().GetFirst(), alpha_hist.GetXaxis().GetLast()+1):
                    alpha_hist[binx, biny] = interp_fit(output_xbins[binx-1], output_ybins[biny-1])

        else:
            print 'Something bad happened.'
            sys.exit()

        alpha_hist.Write()
        print '\n%s written to %s.root' % (hname, fname)




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
        write_alphas_to_root( fname=out_fname, medians=medians, errors=median_errors, xbins=mthad_bins, output_xbins=mthad_out_bins, hname='%s_All_%s' % (hvar.split('/')[0], fit_degree) )
            ## mtt range binned
        write_alphas_to_root( fname=out_fname, medians=binned_mtt_medians, errors=binned_mtt_errors, xbins=mthad_bins, ybins=mtt_bins, output_xbins=mthad_out_bins, output_ybins=mtt_out_bins, hname='%s_Mtt_%s' % (hvar.split('/')[0], fit_degree) )

    #set_trace()

