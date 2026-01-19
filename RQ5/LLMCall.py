from openai import OpenAI
import json
import csv
import os
import re
import time
import pickle
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import csv
import numpy as np
import json
modelname="gpt-4.1"
lang="Python"
client = OpenAI(
    base_url="",
    api_key=""
)

functioncode=None
def call_llm(F1, F2, num_samples=1):
    global  functioncode

    Fun1=functioncode[F1]
    Fun2=functioncode[F2]
    sys_prompt="Please analyze the following two code snippets and determine if they are code clones. Respond with `yes' if the code snippets are clones or `no' if not."
    prompt=f"function snippets:\nFun1:\n{Fun1}\nFun2:\n{Fun2}" 
    try:
        response = client.chat.completions.create(
            model=modelname,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False,  
            temperature=0.0,
            n=num_samples, 

        )

        code_results = []
        for choice in response.choices:
            code = choice.message.content.strip()
            code_results.append(code)
        return code_results[0]
    
    except Exception as e:
        print(f"LLM :{str(e)}")
        return []
def process_chunk(chunk):
    F1,F2,Type=chunk

    result = call_llm(F1,F2)

    return [F1,F2,Type,result]


def init_model():
    global functioncode
    with open('dataset/cross/funs.json', 'r') as f:
        functioncode = json.load(f)

def main():
    init_model ()
    processed=set()
    if os.path.exists(f"{modelname}-{lang}.csv"):
        print(modelname)
        with open(f"{modelname}-{lang}.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                processed.add((row[0],row[1]))
    else:
        with open(f"{modelname}-{lang}.csv", 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["F1","F2","type","result"])
    print(f"Already processed {len(processed)} pairs.")
    work=[]
    with open(f"dataset/cross/{lang}500noclone.csv", 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row_idx, row in enumerate(reader):
            if (row[0],row[1]) in processed:
                continue
            work.append((row[0],row[1],"no"))
    with open(f"dataset/cross/{lang}500.csv", 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row_idx, row in enumerate(reader):
            if (row[0],row[1]) in processed:
                continue
            work.append((row[0],row[1],"T4"))
 


    q = queue.Queue()
    for k in work:
        q.put(k)

    total_tasks = q.qsize()
    print(f"Total tasks: {total_tasks}")
    # finalresults=[]
    with tqdm(total=total_tasks) as pbar:
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_chunk, q.get()) for _ in range(total_tasks)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result[3]!=[]:
                        with open(f"{modelname}-{lang}.csv", 'a', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([result[0], result[1], result[2],result[3]])
                except Exception as e:
                    print(f"Error getting result: {e}")
                pbar.update(1)
if __name__ == '__main__':
    main()
