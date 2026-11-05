'''
Convert a function into a long sequence and compute similarity.
Calls getSimilarity directly.
Input: dot_file_path1, dot_file_path2, bool (whether to normalize)
Output: similarity score between 0 and 1
'''

from xmlrpc.client import boolean
import networkx as nx
import os
# import ssdeep
from simhash import Simhash
import javalang
import xlwt
import sys
import pickle
from detectpath import BASE_PATH

def getGraph(dot_path):
    return nx.DiGraph(nx.nx_pydot.read_dot(dot_path))

def get_simhash(text):
    # generate simhash fingerprint
    return str(Simhash(text).value)

def normalizingLine(s): # e.g. "30:  Attribute createAttribute(KeyValuePair kvp)"
    try:
        tokens = list(javalang.tokenizer.tokenize(s[1:-1]))
        temp_line = []   
        for i in range(len(tokens)):
            token = tokens[i]
            # if (type(token) == javalang.tokenizer.Identifier):
            #     temp_line += 'id'
            # else:
            #     temp_line += token.value
            
            # temp_line.append(ssdeep.hash(str(type(token)).split('.')[-1][:-2])) # -2 to strip trailing ['> ]
            token_type = str(type(token)).split('.')[-1][:-2]
            temp_line.append(get_simhash(token_type))
        return temp_line
    except:
        # print(s[1:-1], "can't parse!!!")
        # return [ssdeep.hash(s[1:-1])]
        return [get_simhash(s[1:-1])]

def commonLine(s): # no normalization
    try:
        tokens = list(javalang.tokenizer.tokenize(s[1:-1])) # strip surrounding quotes
        temp_line = []   
        for i in range(len(tokens)):
            token = tokens[i]
            # if (type(token) == javalang.tokenizer.Identifier):
            #     temp_line += 'id'
            # else:
            #     temp_line += token.value
            # temp_line.append(ssdeep.hash(token.value))
            temp_line.append(get_simhash(token.value))
        return temp_line
    except:
        # print(s[1:-1], "can't parse!!!")
        # return [ssdeep.hash(s[1:-1])]
        return [get_simhash(s[1:-1])]

def get_root_node(G):
    node = None
    for n in G.nodes(data=True):
        predecessors = G.predecessors(n[0])
        if len(list(predecessors)) == 0:
            node = n
            break
    return node[0]


#  takes a digraph as input
def getPreOrder(dg, isNorm : bool):
    stack = []
    result = []
    vis = {} # visited nodes to avoid infinite loops
    root = get_root_node(dg)
    stack.append(root)
    while stack:
        curr = stack.pop()
        vis[curr] = True
        # print(dg.nodes[curr]['label'])
        if (isNorm):
            line = normalizingLine(dg.nodes[curr]['label'])
        else:
            line = commonLine(dg.nodes[curr]['label'])
        result.extend(line)
        for nbr in list(dg[curr])[::-1]:
            if(nbr not in vis):
                stack.append(nbr)
    return result

# Levenshtein edit distance
def editDistance(word1, word2):
    n = len(word1)
    m = len(word2)
    
    # one of the strings is empty
    if n * m == 0:
        return n + m

    # DP array
    D = [ [0] * (m + 1) for _ in range(n + 1)]

    # initialize boundary states
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    # compute all DP values
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                # left_down += 1-ssdeep.compare(word1[i-1], word2[j-1])/100
                hamming_distance = Simhash(int(word1[i-1])).distance(Simhash(int(word2[j-1])))
                # convert hamming distance to similarity in [0, 1]
                similarity = 1 - (hamming_distance / 64.0)  # Simhash uses 64 bits
                left_down += 1 - similarity
            D[i][j] = min(left, down, left_down)
    
    return D[n][m]

 
def getSimilarity(dot_path1, dot_path2, isNorm: bool):
    dg1 = getGraph(dot_path1)
    dg2 = getGraph(dot_path2)
    preOrder1 = getPreOrder(dg1, isNorm)
    preOrder2 = getPreOrder(dg2, isNorm)
    # file = open('size/c3.txt','ab')
    # pickle.dump((preOrder1,preOrder2),file)
    # file.close()
    # size_a = sys.getsizeof(preOrder1)
    # size_b = sys.getsizeof(preOrder2)
    # size = size_a + size_b
    edit_dis = editDistance(preOrder1, preOrder2)
    return (1 - edit_dis/max(len(preOrder1), len(preOrder2)))#,size

def ATVHunter2(sourcefile1,sourcefile2):
    # dotfile1 = './cfg-dot/'+ sourcefile1.split('/')[-1].split('.')[0] + '.dot'
    # dotfile2 = './cfg-dot/'+ sourcefile2.split('/')[-1].split('.')[0] + '.dot'

    file1name = sourcefile1.split('/')[-1]
    file2name = sourcefile2.split('/')[-1]
    #print(file1name)
    cfg1name = file1name.split('.')[0] + '.dot'
    cfg2name = file2name.split('.')[0] + '.dot'

    # extract directory and filename parts
    dir_part = os.path.dirname(BASE_PATH)
    file_part = os.path.basename(BASE_PATH)

    # replace path component to locate cfg-dot directory
    if "prompt" in file_part:
        new_file_part = file_part.replace("prompt", "cfg-dot")
    else:
        # if filename does not contain "prompt", prepend "cfg-dot_"
        new_file_part = "cfg-dot_" + file_part

    # build dot file path
    dotdict = os.path.join(dir_part, new_file_part)


    dotfile1 = os.path.join(dotdict, cfg1name)
    dotfile2 = os.path.join(dotdict, cfg2name)

    return getSimilarity(dotfile1,dotfile2,True)

if __name__ == '__main':
    root_path = 'progex/dot/'
    dot_files = os.listdir('progex/dot/')
    num_of_file = len(dot_files)
    s = [[0 for _ in range(num_of_file)] for _ in range(num_of_file)]
    for i in range(num_of_file):
        for j in range(i+1, num_of_file):
            sim = getSimilarity(os.path.join(root_path ,dot_files[i]), os.path.join(root_path ,dot_files[j]), False)
            s[i][j] = sim
            # print(sim)

    workbook = xlwt.Workbook(encoding = 'utf-8')        # create workbook with utf-8 encoding
    worksheet1 = workbook.add_sheet("filter_sheet")        # add sheet
    worksheet2 = workbook.add_sheet("verify_sheet")

    for i in range(num_of_file):
        worksheet1.write(0,i,dot_files[i])
        worksheet2.write(0,i,dot_files[i])

    for i in range(len(s)):
        for j in range(len(s[i])) :                            
            worksheet1.write(i+1,j,s[i][j])
            worksheet2.write(i+1,j,s[i][j])
    workbook.save('ATVHunter_not_norm_inline.xls') # must use .xls format, not .xlsx


