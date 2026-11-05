import javalang
import os
import re
import sys
import pickle
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


def get_similarity(block1_path, block2_path):
    block1 = getCodeBlock(block1_path)
    block2 = getCodeBlock(block2_path)
    m, n = len(block1), len(block2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if block1[i - 1] == block2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]/max(m, n)

def NiCad_Similarity(f1,f2):
    return get_similarity(f1, f2)
