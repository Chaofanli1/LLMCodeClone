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

def get_sim(tool,dataframe):
    sim = []
    for _,pair in dataframe.iterrows():
        id1,id2 = pair.FunID1,pair.FunID2
        if tool[0] == 'c':               # CFG tool; skip files that failed to parse
            if str(id1) in wrongfile or str(id2) in wrongfile:
                sim.append('parse_errpr')
                continue
        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            # if tool == 'p1':
            #     if id1 == 11673906 and id2 == 11673911:
            #         similarity = 0.9547432550043516
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log ="\n" +  time.asctime() + "\t"+ tool + "\t" + str(id1) + "\t"+ str(id2) + "\t"+ similarity
            logfile.writelines(log)
            similarity = 'False'
        print(similarity)
        sim.append(similarity)

    return sim


def get_simforATV(tool,dataframe):
    sim = []
    for _,pair in dataframe.iterrows():
        id1,id2 = pair.FunID1,pair.FunID2
        if tool[0] == 'c':               # CFG tool; skip files that failed to parse
            if str(id1) in wrongfile or str(id2) in wrongfile:
                sim.append('False')
                continue
        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        similarity = runner(tool, sourcefile1, sourcefile2)
        print(similarity)
        sim.append(similarity)

    return sim


def get_simfornil(dataframe):
    sim = []

    for _,pair in dataframe.iterrows():
        id1,id2 = pair.FunID1,pair.FunID2
        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            similarity = runner('t4', sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log ="\n" +  time.asctime() + "\t"+ 't4' + "\t" + str(id1) + "\t"+ str(id2) + "\t"+ similarity
            logfile.writelines(log)
            similarity = 'False'
        sim.append(similarity)

    return sim


def addcsvcolumns(srccsv, newdata):
    data = pd.read_csv(srccsv)
    print(data.columns)  # print column index values
    #data1 = data['flow']  # get column named 'flow' as new column data
    data['c3_sim'] = newdata  # set new column name to c3_sim
    # data.to_csv(r'new.csv', mode='a', index=False)
    # # mode=a: append mode; header=column names (default true); index=row names (default true); not needed on re-write
    # print(data)

    # remove rows with FALSE
    # for _, pair in data.iterrows():
    #     pair = list(pair)
    #     if 'False' in pair:
    #         data = data.drop(labels=_)
    # print(data)
    data.to_csv('nonew.csv', index=False)

def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num/n))
    df_split = []
    for index in range(n):
        if index < n-1:
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


    #methodtype = 't1'
    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)


    pairs = pairs.iloc[:, :2]
    pairs.columns = ['FunID1','FunID2']

    df_split = cut_df(pairs, 60)

    func = partial(get_sim, 'a1')
    pool = mp.Pool(processes=60)
    sim_a1 = []
    it_sim_a1 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_a1:
        sim_a1 = sim_a1 + item
    pool.close()
    pool.join()
    
    
    func = partial(get_sim, 'a2')
    pool = mp.Pool(processes=60)
    sim_a2 = []
    it_sim_a2 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_a2:
        sim_a2 = sim_a2 + item
    pool.close()
    pool.join()
    
    func = partial(get_sim, 'a3')
    pool = mp.Pool(processes=60)
    sim_a3 = []
    it_sim_a3 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_a3:
        sim_a3 = sim_a3 + item
    pool.close()
    pool.join()
    
    func = partial(get_sim, 't1')
    pool = mp.Pool(processes=60)
    sim_t1 = []
    it_sim_t1 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_t1:
        sim_t1 = sim_t1 + item
    pool.close()
    pool.join()
    
    func = partial(get_sim, 't2')
    pool = mp.Pool(processes=60)
    sim_t2 = []
    it_sim_t2 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_t2:
        sim_t2 = sim_t2 + item
    pool.close()
    pool.join()
    
    func = partial(get_sim, 't3')
    pool = mp.Pool(processes=60)
    sim_t3 = []
    it_sim_t3 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_t3:
        sim_t3 = sim_t3 + item
    pool.close()
    pool.join()
    
    # pool = mp.Pool(processes=60)
    # it_sim_t4 = tqdm.tqdm(pool.imap(get_simfornil, df_split))
    # sim_t4 = []
    # for i in it_sim_t4:
    #     sim_t4 = sim_t4 + i
    # pool.close()
    # pool.join()
    
    func = partial(get_sim, 'c1')
    pool = mp.Pool(processes=60)
    sim_c1 = []
    it_sim_c1 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_c1:
        sim_c1 = sim_c1 + item
    pool.close()
    pool.join()
    
    func = partial(get_sim, 'c2')
    pool = mp.Pool(processes=60)
    sim_c2 = []
    it_sim_c2 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_c2:
        sim_c2 = sim_c2 + item
    pool.close()
    pool.join()

    func = partial(get_sim, 'c3')
    pool = mp.Pool(processes=60)
    sim_c3 = []
    it_sim_c3 = tqdm.tqdm(pool.imap(func, df_split))
    for item in it_sim_c3:
        sim_c3 = sim_c3 + item
    pool.close()
    pool.join()

    result = pd.DataFrame({'FunID1': pairs['FunID1'].to_list(), 'FunID2': pairs['FunID2'].to_list(),
                           't1_sim': sim_t1, 't2_sim': sim_t2, 't3_sim': sim_t3,
                           'a1_sim': sim_a1, 'a2_sim': sim_a2, 'a3_sim': sim_a3, 
                           'c1_sim': sim_c1, 'c2_sim': sim_c2,'c3_sim':sim_c3})

    # result.to_csv('test.csv', index=False)
    result.to_csv(outputdir + Clonetype + '_sim.csv', index=False)


if __name__ == '__main__':
    main()
