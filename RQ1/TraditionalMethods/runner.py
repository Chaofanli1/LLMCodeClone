import os
import sys
import getopt
from AST_AST2014 import AST2014
from AST_BWCCA2015 import BWCCA2015
from AST_COMPSAC2018 import COMPSAC2018
from Token_NiCad import NiCad_Similarity
# from Token_NIL import NIL_Similarity
from Token_LVMapper import LVMapper_Similarity
from Token_SourcererCC import SourcererCC_Similarity
from CFG_StoneDetector import StoneDetector
from CFG_Centroids import Centroids
from CFG_ATVHunter2 import ATVHunter2
# from PDG_CCGraph import CCGraph
import faulthandler
def main(argv):
    tool = ''
    try:
        opts, args = getopt.getopt(argv, "ht:", ["tool="])
    except getopt.GetoptError:
        print('runner.py -t <tools> <sourcefile1> <sourcefile2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('runner.py -t <tools> <sourcefile1> <sourcefile2>\n\
            a1--AST2014 a2--BWCCA2015 a3--COMPSAC2018\n\
            t1--NiCad t2--SourcererCC t3--LVMapper\n\
            c1--ICSME2021 c2--SSEPRW2017\n\
            p1--JSS2022')
            sys.exit()
        elif opt in ("-t", "--tool"):
            tool = arg
    if len(args) < 2:
        print('runner.py -t <tools> <src1> <src2>')

    return tool, args[0], args[1]

def runner(tool, sourcefile1, sourcefile2):
    # ast
    if tool == 'a1':
        return AST2014(sourcefile1, sourcefile2)

    elif tool == 'a2':
        return BWCCA2015(sourcefile1, sourcefile2)

    elif tool == 'a3':
        return COMPSAC2018(sourcefile1, sourcefile2)

    # token
    if tool == 't1':
        return NiCad_Similarity(sourcefile1, sourcefile2)

    elif tool == 't2':
        return SourcererCC_Similarity(sourcefile1, sourcefile2)

    elif tool == 't3':
        return LVMapper_Similarity(sourcefile1, sourcefile2)
    # elif tool == 't4':
    #     return NIL_Similarity(sourcefile1, sourcefile2)

    # cfg
    elif tool == 'c1':
        return StoneDetector(sourcefile1, sourcefile2, 0)
    elif tool == 'c2':
        return Centroids(sourcefile1, sourcefile2)
    elif tool =='c3':
        return ATVHunter2(sourcefile1,sourcefile2)
    # elif tool =='p1':
    #     return CCGraph(sourcefile1,sourcefile2)

if __name__ == "__main__":
    faulthandler.enable()
