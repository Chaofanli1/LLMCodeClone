import os
import sys
import getopt
from Token_NiCad import NiCad_Similarity
from Token_LVMapper import LVMapper_Similarity
from Token_SourcererCC import SourcererCC_Similarity
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
            t1--NiCad t2--SourcererCC t3--LVMapper')
            sys.exit()
        elif opt in ("-t", "--tool"):
            tool = arg
    if len(args) < 2:
        print('runner.py -t <tools> <src1> <src2>')

    return tool, args[0], args[1]

def runner(tool, sourcefile1, sourcefile2):
    # token
    if tool == 't1':
        return NiCad_Similarity(sourcefile1, sourcefile2)

    elif tool == 't2':
        return SourcererCC_Similarity(sourcefile1, sourcefile2)

    elif tool == 't3':
        return LVMapper_Similarity(sourcefile1, sourcefile2)

if __name__ == "__main__":
    faulthandler.enable()
