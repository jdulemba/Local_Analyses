from pdb import set_trace
import rootpy.plotting as plotting
from rootpy.plotting import views, Graph, Hist, Hist2D
from rootpy import asrootpy
import numpy as np
from RebinView import RebinView

def stack_plots(lists):

    lists.sort(key=lambda x: x.Integral())
    total = 0
    ratio_hists = []
    Stack_hists = []

    for i in lists:
        total += i
    for i in lists:
        ratio_hists.append(i/total)
        #i.SetFillStyle(1001)
        Stack_hists.append(i)

    stack = plotting.HistStack()
    norm_stack = plotting.HistStack()

#    print 'stack plots'
#    set_trace()

    for i in Stack_hists:
        stack.Add(i)
        norm_stack.Add(i/total)


    return stack, norm_stack, ratio_hists




def FindMedianAndMedianError(h1d):

    if h1d.DIM != 1:
        print 'Histogram needs to be 1D to find median!'

        median, median_error = -10., -10.
        return median, median_error

    if h1d.Integral() == 0:
        median, median_error = -10., -10.
        return median, median_error

    probs = np.array([h1d.Integral(1, i)/h1d.Integral() for i in range(1, h1d.GetNbinsX())])
    median = h1d.GetBinCenter(np.where(probs>= 0.5)[0][0])
    median_error = 1.2533*h1d.GetMeanError() if h1d.GetMeanError() != 0 else 1e-4 #standard error of median

    return median, median_error


def median_from_3d_hist(h3d, projection, xbins, ybins):

    medians = []
    median_errors = []

    #set_trace()
    h2d = asrootpy(h3d.Project3D(projection)) # projection formatted as 'new_yaxis new_xaxis' ('zx' for example)
    h2d = RebinView.newRebin2D(h2d, xbins, ybins)
    h2d.xaxis.range_user = min(xbins), max(xbins)

    for xbin in range(1, h2d.GetNbinsX()+1):
        hist_yproj = asrootpy(h2d.ProjectionY("", xbin, xbin).Clone())

        median, median_error = FindMedianAndMedianError(hist_yproj)
        medians.append(median)
        median_errors.append(median_error)

    return medians, median_errors

