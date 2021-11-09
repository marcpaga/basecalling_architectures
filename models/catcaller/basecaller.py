"""This script can be used to basecall a set of fast5 files given a model,
it will produce one (or more) fasta files with the basecalls
"""

import sys
sys.path.append('/hpc/compgen/users/mpages/babe/src')

from classes import BaseBasecaller

class Basecaller(BaseBasecaller):
    """
    """
    
    def __init__(self, *args, **kwargs):
        super(Basecaller, self).__init__(*args, **kwargs)
        
    def stich(self, *args, **kwargs):
        return self.stitch_by_stride(*args, **kwargs)
    
    