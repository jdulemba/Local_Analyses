import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from argparse import ArgumentParser
import Utilities.python.functions as fncts
import os, math

parser = ArgumentParser()
parser.add_argument('year', choices=['2016', '2017', '2018'], help='choose year to compare nanoAOD output to miniAOD')
args = parser.parse_args()

basedir = '%s/nanoAOD/plots/%s' % (os.environ['ANALYSES_PROJECT'], args.year)
if not os.path.isdir(basedir):
    os.makedirs(basedir)

fdir = '/Users/jdulemba/RESEARCH/NTuples/2016Legacy/root_testing' if args.year == '2016' else '/Users/jdulemba/RESEARCH/NTuples/%s/root_testing' % args.year
    ## open nanoAOD file and get events
nanofname = 'ttJets_2016NanoAOD.root' if args.year == '2016' else 'ttJetsSL_%sNanoAOD.root' % args.year
nanofile = uproot.open("%s/%s" % (fdir, nanofname))
nanoEvts = nanofile['Events']

    ## open miniAOD file and get events
minifname = 'ttJets_2016miniAODskimmed.root' if args.year == '2016' else 'ttJetsSL_%sminiAODskimmed.root' % args.year
minifile = uproot.open("%s/%s" % (fdir, minifname))
miniEvts = minifile['Events']

    ## make dataframes
nano_df = nanoEvts.pandas.df(["luminosityBlock", "event", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_isGlobal", 'Muon_isPFcand', 'Muon_nStations', 'Muon_dxy', 'Muon_dz', 'Muon_nTrackerLayers', "Muon_tightId", "Muon_isTracker"])
mini_df = miniEvts.pandas.df(["lumi", "evt", "muons.pt", "muons.eta", "muons.phi", "muons.isGlobal", "muons.isPF", "muons.numMatchedStations", "muons.dxy", "muons.dz", "muons.trackerLayers", "muons.chi2", "muons.ndof", "muons.validHits", "muons.pixelHits"])

    ## get intersection of events
nano_evts = nano_df.event.unique()
nano_lumis = nano_df.luminosityBlock.unique()
mini_evts = mini_df.evt.unique()
mini_lumis = mini_df.lumi.unique()

lumi_mask = np.intersect1d(nano_lumis, mini_lumis)
evt_mask = np.intersect1d(nano_evts, mini_evts) 


    ## make skimmed dataframes
nano_skim = nano_df[nano_df.event.isin(evt_mask)]
mini_skim = mini_df[mini_df.evt.isin(evt_mask)]

    ## sort dataframes by event number
nano_skim = nano_skim.sort_values('event')
mini_skim = mini_skim.sort_values('evt')

    ## compute tightID for miniAOD
mini_skim['isTight'] = (mini_skim.get('muons.isGlobal')) & (mini_skim.get('muons.isPF')) & (mini_skim.get('muons.chi2')/mini_skim.get('muons.ndof') < 10.)\
        & (mini_skim.get('muons.validHits') > 0) & (mini_skim.get('muons.numMatchedStations') > 1) & (abs(mini_skim.get('muons.dxy')) < 0.2)\
        & (abs(mini_skim.get('muons.dz')) < 0.5) & (mini_skim.get('muons.pixelHits') > 0) & (mini_skim.get('muons.trackerLayers') > 5.)

    ## skim nanoAOD events based on same cuts as applied when skimming miniAOD in PATTools/python/objects/muons.py
nano_skim = nano_skim[(nano_skim.Muon_pt > 10.) & (abs(nano_skim.Muon_eta) < 2.5) & (nano_skim.Muon_isGlobal | nano_skim.Muon_isTracker)]

    # filter events that aren't in both dataframes after skimming
nano_evts_skim = nano_skim.event.unique()
mini_evts_skim = mini_skim.evt.unique()

diff_evts = np.setdiff1d(nano_evts_skim, mini_evts_skim)
same_evts = np.intersect1d(nano_evts_skim, mini_evts_skim)
same_evts_list = list(same_evts)

    ## skim dataframes again
nano_skim = nano_skim[nano_skim.event.isin(same_evts)]
mini_skim = mini_skim[mini_skim.evt.isin(same_evts)]

diff_inds = []
diff_tightID = []

report = int(math.pow(10, math.floor(math.log10(len(same_evts_list)))-1))
print 'Reporting every %i events\n' % report
#set_trace()

    ## find differences in events
for idx, evt in enumerate(same_evts):
    nano_evt = nano_skim[nano_skim.event == evt]
    mini_evt = mini_skim[mini_skim.evt == evt]

        ## sort by muon pt (hopefully gets rid of flipped subentries
    nano_evt = nano_evt.sort_values('Muon_pt', ascending=False)
    mini_evt = mini_evt.sort_values('muons.pt', ascending=False)

    if nano_evt.shape[0] != mini_evt.shape[0]:
        #print 'Diff dimensions: %i' % evt
        diff_inds.append(evt)
    elif not np.array_equal(nano_evt.Muon_tightId.values, mini_evt.isTight.values):
        print 'isTight different for %i' % evt
        diff_tightID.append(evt)
        #set_trace()

    if (idx+1) % report == 0:
        print '%i/%i events processed' % (idx+1, len(same_evts_list))

#set_trace()

    ## plot miniAOD variables for events with different tightID results
mini_vars_of_interest = [key for key in miniEvts.keys() if ('muons' in key or 'vertex' in key)]
for evt in diff_tightID:
    for var in mini_vars_of_interest:
        df = miniEvts.pandas.df(["evt", var])
        val = df[df.evt == evt].get(var).values

        #set_trace()
        fig = plt.figure()
        _ = plt.hist(val, bins='auto')
        plt.title("miniAOD event %i" % evt)
        plt.xlabel('%s = %f' % (var, val[0]) if val.size == 1 else var)
        figname = 'miniAOD_evt%i_%s' % (evt, var.replace('.', '_'))
        figdir = '/'.join([basedir, 'diff_muID', 'evt%i' % evt, 'miniAOD'])
        if not os.path.isdir(figdir):
            os.makedirs(figdir)
        fig.savefig('%s/%s' % (figdir, figname))
        print '%s created' % figname
        plt.close()        


    ## plot nanoAOD variables for events with different tightID results
nano_vars_of_interest = [key for key in nanoEvts.keys() if (('Muon' in key or 'PV' in key) and 'HLT' not in key )]
for evt in diff_tightID:
    for var in nano_vars_of_interest:
        df = nanoEvts.pandas.df(["event", var])
        val = df[df.event == evt].get(var).values

        #set_trace()
        fig = plt.figure()
        _ = plt.hist(val, bins='auto')
        plt.title("nanoAOD event %i" % evt)
        plt.xlabel('%s = %f' % (var, val[0]) if val.size == 1 else var)
        figname = 'nanoAOD_evt%i_%s' % (evt, var)
        figdir = '/'.join([basedir, 'diff_muID', 'evt%i' % evt, 'nanoAOD'])
        if not os.path.isdir(figdir):
            os.makedirs(figdir)
        fig.savefig('%s/%s' % (figdir, figname))
        print '%s created' % figname
        plt.close()        

