# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:49:43 2019

@author: bchassagno
"""

import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

bn=gum.fastBN("A->B")
gnb.showBN(bn,size="15")