'''

View to rebin a histogram.

Author: Evan K. Friis, UW Madison

'''

import array
import rootpy.plotting.views as views
import rootpy.plotting as plotting
import rootpy
try:
    from rootpy.utils import asrootpy
except ImportError:
    from rootpy import asrootpy
import ROOT
import os
from quad import quad
from pdb import set_trace
log = rootpy.log["/RebinView"]
ROOT.TH1.SetDefaultSumw2(True)

class RebinView(views._FolderView):
    ''' Rebin a histogram.

    The original histogram is unmodified, a rebinned clone is returned.

    '''
    def __init__(self, dir, binning):
        self.binning = binning
        super(RebinView, self).__init__(dir)

    @staticmethod
    def newRebin3D(histogram, bin_arrayx, bin_arrayy, bin_arrayz):
        'Rebin 3D histo with irregular bin size'

        #old binning
        oldbinx = [float(histogram.GetXaxis().GetBinLowEdge(1))]
        oldbiny = [float(histogram.GetYaxis().GetBinLowEdge(1))]
        oldbinz = [float(histogram.GetZaxis().GetBinLowEdge(1))]
        oldbinx.extend(float(histogram.GetXaxis().GetBinUpEdge(x)) for x in xrange(1, histogram.GetNbinsX()+1))
        oldbiny.extend(float(histogram.GetYaxis().GetBinUpEdge(y)) for y in xrange(1, histogram.GetNbinsY()+1))
        oldbinz.extend(float(histogram.GetZaxis().GetBinUpEdge(z)) for z in xrange(1, histogram.GetNbinsZ()+1))
        
        #if new binninf is just one number and int, use it to rebin rather than as edges
        if len(bin_arrayx) == 1 and isinstance(bin_arrayx[0], int):
            nrebin = bin_arrayx[0]
            bin_arrayx = [j for i, j in enumerate(oldbinx) if i % nrebin == 0]
        if len(bin_arrayy) == 1 and isinstance(bin_arrayy[0], int):
            nrebin = bin_arrayy[0]
            bin_arrayy = [j for i, j in enumerate(oldbiny) if i % nrebin == 0]
        if len(bin_arrayz) == 1 and isinstance(bin_arrayz[0], int):
            nrebin = bin_arrayz[0]
            bin_arrayz = [j for i, j in enumerate(oldbinz) if i % nrebin == 0]

        #create a clone with proper binning
        # from pdb import set_trace; set_trace()
        new_histo = plotting.Hist3D(
            bin_arrayx,
            bin_arrayy,
            bin_arrayz,
            #name = histogram.name,
            title = histogram.title,
            **histogram.decorators
        )
        new_histo.xaxis.title = histogram.xaxis.title
        new_histo.yaxis.title = histogram.yaxis.title
        new_histo.zaxis.title = histogram.zaxis.title

        #check that new bins don't overlap on old edges
        for x in bin_arrayx:
            if x==0:
                if not any( abs((oldx)) < 10**-8 for oldx in oldbinx ):
                    raise Exception('New bin edge in x axis %s does not match any old bin edge, operation not permitted' % x)
            else:
                if not any( abs((oldx / x)-1.) < 10**-8 for oldx in oldbinx ):
                    raise Exception('New bin edge in x axis %s does not match any old bin edge, operation not permitted' % x)
        for y in bin_arrayy:
            if y ==0:
                if not any( abs((oldy) )< 10**-8 for oldy in oldbiny ):
                    raise Exception('New bin edge in y axis %s does not match any old bin edge, operation not permitted' % y)
            else:
                if not any( abs((oldy / y)-1.) < 10**-8 for oldy in oldbiny ):
                    raise Exception('New bin edge in y axis %s does not match any old bin edge, operation not permitted' % y)
        for z in bin_arrayz:
            if z ==0:
                if not any( abs((oldz) )< 10**-8 for oldz in oldbinz ):
                    raise Exception('New bin edge in z axis %s does not match any old bin edge, operation not permitted' % z)
            else:
                if not any( abs((oldz / z)-1.) < 10**-8 for oldz in oldbinz ):
                    raise Exception('New bin edge in z axis %s does not match any old bin edge, operation not permitted' % z)
        
        #fill the new histogram
        for x in xrange(0, histogram.GetNbinsX()+2 ):
            for y in xrange(0, histogram.GetNbinsY()+2 ):
                for z in xrange(0, histogram.GetNbinsZ()+2 ):
                    new_bin_x = new_histo.GetXaxis().FindFixBin(
                        histogram.GetXaxis().GetBinCenter(x)
                        )
                    new_bin_y = new_histo.GetYaxis().FindFixBin(
                        histogram.GetYaxis().GetBinCenter(y)
                        )
                    new_bin_z = new_histo.GetZaxis().FindFixBin(
                        histogram.GetZaxis().GetBinCenter(z)
                        )
                    new_histo.SetBinContent(
                        new_bin_x, new_bin_y, new_bin_z,
                        histogram.GetBinContent(x,y,z) + 
                        new_histo.GetBinContent(new_bin_x, new_bin_y, new_bin_z)
                        )
                    new_histo.SetBinError(
                        new_bin_x, new_bin_y, new_bin_z,
                        quad(
                            histogram.GetBinError(x,y,z),
                            new_histo.GetBinError(new_bin_x, new_bin_y, new_bin_z)
                            )
                        )
                    #new_histo.Fill(
                    #    histogram.GetXaxis().GetBinCenter(x), 
                    #    histogram.GetYaxis().GetBinCenter(y), 
                    #    histogram.GetBinContent(x,y)
                    #    )

        new_histo.SetEntries( histogram.GetEntries() )
        return new_histo
                              
    @staticmethod
    def newRebin2D(histogram, bin_arrayx, bin_arrayy):
        'Rebin 2D histo with irregular bin size'

        #old binning
        oldbinx = [float(histogram.GetXaxis().GetBinLowEdge(1))]
        oldbiny = [float(histogram.GetYaxis().GetBinLowEdge(1))]
        oldbinx.extend(float(histogram.GetXaxis().GetBinUpEdge(x)) for x in xrange(1, histogram.GetNbinsX()+1))
        oldbiny.extend(float(histogram.GetYaxis().GetBinUpEdge(y)) for y in xrange(1, histogram.GetNbinsY()+1))
        
        #if new binninf is just one number and int, use it to rebin rather than as edges
        if len(bin_arrayx) == 1 and isinstance(bin_arrayx[0], int):
            nrebin = bin_arrayx[0]
            bin_arrayx = [j for i, j in enumerate(oldbinx) if i % nrebin == 0]
        if len(bin_arrayy) == 1 and isinstance(bin_arrayy[0], int):
            nrebin = bin_arrayy[0]
            bin_arrayy = [j for i, j in enumerate(oldbiny) if i % nrebin == 0]

        #create a clone with proper binning
        # from pdb import set_trace; set_trace()
        new_histo = plotting.Hist2D(
            bin_arrayx,
            bin_arrayy,
            #name = histogram.name,
            title = histogram.title,
            **histogram.decorators
        )
        new_histo.xaxis.title = histogram.xaxis.title
        new_histo.yaxis.title = histogram.yaxis.title

        #check that new bins don't overlap on old edges
        for x in bin_arrayx:
            if x==0:
                if not any( abs((oldx)) < 10**-8 for oldx in oldbinx ):
                    raise Exception('New bin edge in x axis %s does not match any old bin edge, operation not permitted' % x)
            else:
                if not any( abs((oldx / x)-1.) < 10**-8 for oldx in oldbinx ):
                    raise Exception('New bin edge in x axis %s does not match any old bin edge, operation not permitted' % x)
        for y in bin_arrayy:
            if y ==0:
                if not any( abs((oldy) )< 10**-8 for oldy in oldbiny ):
                    raise Exception('New bin edge in y axis %s does not match any old bin edge, operation not permitted' % y)
            else:
                if not any( abs((oldy / y)-1.) < 10**-8 for oldy in oldbiny ):
                    raise Exception('New bin edge in y axis %s does not match any old bin edge, operation not permitted' % y)
        
        #fill the new histogram
        for x in xrange(0, histogram.GetNbinsX()+2 ):
            for y in xrange(0, histogram.GetNbinsY()+2 ):
                new_bin_x = new_histo.GetXaxis().FindFixBin(
                    histogram.GetXaxis().GetBinCenter(x)
                    )
                new_bin_y = new_histo.GetYaxis().FindFixBin(
                    histogram.GetYaxis().GetBinCenter(y)
                    )
                new_histo.SetBinContent(
                    new_bin_x, new_bin_y,
                    histogram.GetBinContent(x,y) + 
                    new_histo.GetBinContent(new_bin_x, new_bin_y)
                    )
                new_histo.SetBinError(
                    new_bin_x, new_bin_y,
                    quad(
                        histogram.GetBinError(x,y),
                        new_histo.GetBinError(new_bin_x, new_bin_y)
                        )
                    )
                #new_histo.Fill(
                #    histogram.GetXaxis().GetBinCenter(x), 
                #    histogram.GetYaxis().GetBinCenter(y), 
                #    histogram.GetBinContent(x,y)
                #    )

        new_histo.SetEntries( histogram.GetEntries() )
        return new_histo
                              
    @staticmethod
    def rebin(histogram, binning):
        ''' Rebin a histogram

        [binning] can be either an integer, or a list/tuple for variable bin
        sizes.

        '''
        # Just merging bins
        if isinstance(binning, int):
            if binning == 1 or isinstance(histogram, ROOT.TH2):
                return histogram
            histogram.Rebin(binning)
            return histogram
        # Fancy variable size bins
        if isinstance(histogram, ROOT.TH2):
            if not isinstance(binning[0], (list, tuple)):
                return histogram
            #print binning[0], ' ' , binning[1] 
            bin_arrayx = binning[0] #array.array('d',binning[0])
            bin_arrayy = binning[1] #array.array('d',binning[1])
            if len(binning[0]) ==1 and len(binning[1])==1 :                
                return  asrootpy(histogram.Rebin2D(int(binning[0][0]),int(binning[1][0]), histogram.GetName() + 'rebin'))
            else:
                return RebinView.newRebin2D(histogram, bin_arrayx, bin_arrayy)
        elif isinstance(histogram, ROOT.TH1):
            if isinstance(binning[0], (list, tuple)):
                raise ValueError(
                    "You are providing a binning scheme (%s) "
                    "not compatible with 1D hists!" % (binning.__repr__()))
            bin_array = array.array('d', binning)
            ret = asrootpy( histogram.Rebin(len(binning)-1, histogram.GetName() + 'rebin', bin_array) )
            if hasattr(histogram, 'decorators'):
                ret.decorate( **histogram.decorators )
            return ret 
        else:
            raise ValueError('ERROR in RebinView: not a TH1 or TH2 histo. Rebin not done')
            return histogram

    def apply_view(self, object):
        object = object.Clone()
        return RebinView.rebin(object, self.binning)
