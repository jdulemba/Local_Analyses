#! /usr/bin/env python

import sys
import os
from fnmatch import fnmatch
from optparse import OptionParser
import ROOT 
from pdb import set_trace

__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch)"
__doc__ = """script to look at the content of the root file:\n\n
Usage: rootfind file.root [options]"""

def GetContent(dir):
    'does not read key content'
    keys = dir.GetListOfKeys()
    return [(getattr(ROOT, i.GetClassName()), i.GetName(), i.GetTitle() if hasattr(i, 'GetTitle') else '') for i in keys]

def inherits_from(test, target):
    if test == target:
        return True
    return any( inherits_from(test, sub) 
                for sub in target.__subclasses__())

def rootfind( directory, dirName='', **kwargs ):
    if 'max_depth' in kwargs and kwargs['max_depth'] == 0:
        return
    elif 'max_depth' in kwargs:
        kwargs['max_depth'] -= 1
        
    dirContent = GetContent(directory)
    for obj_type, obj_name, _ in dirContent:
      path = os.path.join(dirName,obj_name)
      if 'name' not in kwargs or fnmatch(path,kwargs['name']):
          if 'type' not in kwargs or \
             not kwargs['type'] or \
             inherits_from(obj_type, getattr(ROOT, kwargs['type'])):
              addenda = ''
              if 'code' in kwargs and kwargs['code']:
                  entry = directory.Get(obj_name)
                  for code in  kwargs['code'].split('$'):
                      toeval  = 'entry.%s' % code
                      addenda += '%s ' % eval(toeval)
              print path,addenda
      if inherits_from(obj_type, ROOT.TDirectory):
         subdirName = os.path.join(dirName,obj_name)
         entry= directory.Get(obj_name)
         rootfind(entry, subdirName, **kwargs)

def parse_options(arguments=sys.argv[1:]):
   parser = OptionParser(description=__doc__)
   parser.add_option('--name', metavar='pattern', type=str, default = '*',
                     help='allows the search for a particular pattern',dest='name')
   parser.add_option('--type', metavar='class'  , type=str, dest='type',default = '',
                     help='searches only for objects matching (or inheriting from) a particular class')
   parser.add_option('--exec', metavar='code'   , type=str,  dest='code', default = '',
                     help='executes a class method on each object found (beware of errors!) and returns the result together with the found object path')
   parser.add_option('--show-errors', action='store_true', dest='show_errors', default = False,
                     help='prints out root error messages when opening the file (usually a bit annoying and therefore suppressed by default)')
   parser.add_option('--max-depth', dest='max_depth', default = -1, type=int,
                     help='maximum depth where to search')
   parser.add_option('--title', action='store_true',
                     help='shortcut to --exec="GetTitle()"')
   parser.add_option('--cname', action='store_true',
                     help='shortcut to --exec="ClassName()"')

   return parser.parse_args(args=arguments)
   

if __name__ == '__main__':
    (options,file_names) = parse_options()
    if options.title:
        options.code = 'GetTitle()'
    elif options.cname:
        options.code = 'ClassName()'
        
    tfile_names = [name for name in file_names if '.root' in name]
    
    if not options.show_errors:
        ROOT.gErrorIgnoreLevel = ROOT.kError+1 #Suppress anything that is not an exception

    for name in tfile_names:
        tfile = ROOT.TFile.Open(name)
        print "In %s:" % tfile.GetName()
        rootfind( tfile, **vars(options))
        tfile.Close()
