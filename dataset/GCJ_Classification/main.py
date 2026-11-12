import pandas as pd
import sys
from runner import runner
import time
import multiprocessing as mp
import tqdm
import math
from functools import partial

# ── Configuration ──────────────────────────────────────────────────────────────
# Path to directory containing Java function files
inputpath = '/path/to/java_functions/'
# Path to input clone pairs CSV
INPUT_CSV = "/path/to/input/clone_pairs.csv"
# Output directory for similarity results
OUTPUT_DIR = "/path/to/output/"
# ───────────────────────────────────────────────────────────────────────────────

def get_sim(tool, dataframe):
    sim = []
    for _, pair in dataframe.iterrows():
        id1, id2 = pair.FunID1, pair.FunID2
        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = 'False'
        sim.append(similarity)
    return sim


def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num / n))
    df_split = []
    for index in range(n):
        if index < n - 1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split


def main():
    inputcsv = INPUT_CSV
    outputdir = OUTPUT_DIR
    Clonetype = inputcsv.split('/')[-1].split('.')[0]
    if 'noclone' in inputcsv:
        Clonetype = 'noclone'

    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)
    pairs = pairs.iloc[:, :2]
    pairs.columns = ['FunID1', 'FunID2']

    df_split = cut_df(pairs, 60)

    func = partial(get_sim, 't1')
    pool = mp.Pool(processes=60)
    sim_t1 = []
    for item in tqdm.tqdm(pool.imap(func, df_split)):
        sim_t1 = sim_t1 + item
    pool.close()
    pool.join()

    func = partial(get_sim, 't2')
    pool = mp.Pool(processes=60)
    sim_t2 = []
    for item in tqdm.tqdm(pool.imap(func, df_split)):
        sim_t2 = sim_t2 + item
    pool.close()
    pool.join()

    func = partial(get_sim, 't3')
    pool = mp.Pool(processes=60)
    sim_t3 = []
    for item in tqdm.tqdm(pool.imap(func, df_split)):
        sim_t3 = sim_t3 + item
    pool.close()
    pool.join()

    result = pd.DataFrame({
        'FunID1': pairs['FunID1'].to_list(),
        'FunID2': pairs['FunID2'].to_list(),
        't1_sim': sim_t1,
        't2_sim': sim_t2,
        't3_sim': sim_t3,
    })

    result.to_csv(outputdir + Clonetype + '_sim.csv', index=False)


if __name__ == '__main__':
    main()
