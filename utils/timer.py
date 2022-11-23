"""
Created on Feb 7

@author: lachaji

"""

import time
import datetime
  
def convert(n):
    return str(datetime.timedelta(seconds = n))

class Timer:
    def __init__(self):
        self.start = None
        self.end = None
    
    def tik(self):
        self.start = time.time()
    def tak(self):
        self.end = time.time()
        return convert( self.end - self.start )