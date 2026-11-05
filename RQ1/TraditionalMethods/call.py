from runner import runner
import csv
import os
import re
import time
import pickle
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import numpy as np
import json
import math
from functools import partial

# ── Configuration ──────────────────────────────────────────────────────────────
# Path to directory containing Java function files
FUNS_DIR = '/path/to/java_functions/'
# Path to input no-clone pairs CSV
INPUT_CSV = "/path/to/input/noclone_pairs.csv"
# Output CSV path for no-clone pair similarities
OUTPUT_CSV = "/path/to/output/noclone_results.csv"
# ───────────────────────────────────────────────────────────────────────────────
def process_chunk(chunk):
    F1,F2,Type=chunk
    sourcefile1=f'{FUNS_DIR}{F1}.java'
    sourcefile2=f'{FUNS_DIR}{F2}.java'
    tools=['t1','t2','t3','a1','a2','a3','c1','c2','c3']
    # tools=['c1','c2','c3']
    row=[F1,F2,Type]
    for tool in tools:
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity=False
        row.append(similarity)
    return row
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


def main():
    outputcsv = OUTPUT_CSV
    processed=set()
    if os.path.exists(outputcsv):
        with open(outputcsv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # next(reader)  # skip header
            for row in reader:
                processed.add((row[0],row[1]))
    else:
        with open(outputcsv, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["F1","F2","result"])
    print(f"Already processed {len(processed)} pairs.")
    work=[]
    q = queue.Queue()
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as csvfile:  # input no-clone pairs CSV
        reader = csv.reader(csvfile)
        next(reader)
        for row_idx, row in enumerate(reader):
            if (row[0],row[1]) in processed:
                continue

            work.append((row[0],row[1],row[2]))
    # work.append(("184251","184252","VST3"))
    # work.append(("184289","184290","VST3"))

    for k in work:
        q.put(k)
        # # # break
        if q.qsize()>=1000:
            break

    total_tasks = q.qsize()
    print(f"Total tasks: {total_tasks}")
    finalresults=[]
    with tqdm(total=total_tasks) as pbar:
        with ProcessPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(process_chunk, q.get()) for _ in range(total_tasks)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with open(OUTPUT_CSV, 'a', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(result)
                except Exception as e:
                    print(f"Error getting result: {e}")
                pbar.update(1)
if __name__ == '__main__':
    main()
