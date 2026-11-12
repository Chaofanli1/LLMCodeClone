import javalang
import re
__GloPT__ = dict()


def generate_GPT(code_blocks) :
    # builds global probability table from both code blocks
    for block in code_blocks:
        for token in block:
            if (__GloPT__.__contains__(token)) :
                __GloPT__[token] = __GloPT__[token] + 1
            else:
                __GloPT__[token] = 1

def sortBlockWithGPT(block) :
    # Sort tokens by ascending frequency in GPT
    block.sort(key=lambda item: __GloPT__[item])

def remove_comments(code):
    # Remove block comments and line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code

def getCodeBlock(file_path):
    block = []
    with open(file_path, 'r') as temp_file:
        code = temp_file.read()
        code = remove_comments(code)  # remove comments first
        lines = code.splitlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if (type(token) != javalang.tokenizer.Separator and type(token) != javalang.tokenizer.Operator):
                    block.append(token.value)
    return block

def overlapSimilarity(ls_1, ls_2):
    res = 0
    len1, len2 = len(ls_1), len(ls_2)
    i_1, i_2 = 0, 0
    while (i_1 < len1 and i_2 < len2) :
        if (ls_1[i_1] == ls_2[i_2]) :
            res += 1
            i_1 += 1
            i_2 += 1
        else :
            if (__GloPT__[ls_1[i_1]] < __GloPT__[ls_2[i_2]]) :
                i_1 += 1
            elif (__GloPT__[ls_1[i_1]] > __GloPT__[ls_2[i_2]]):
                i_2 += 1
            else :
                if(ls_1[i_1] < ls_2[i_2]):
                    i_1 += 1
                else :
                    i_2 += 1
    return res

def get_similarity(block1_path, block2_path):
    block1 = getCodeBlock(block1_path)
    block2 = getCodeBlock(block2_path)
    generate_GPT([block1, block2])
    sortBlockWithGPT(block1)
    sortBlockWithGPT(block2)
    lcs_len = overlapSimilarity(block1, block2)
    return lcs_len / max(len(block1), len(block2))

def SourcererCC_Similarity(f1,f2):
    return get_similarity(f1, f2)