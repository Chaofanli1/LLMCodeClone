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
client = OpenAI(
    base_url="",
    api_key=""
)
functioncode=None
def call_llm(F1, F2, num_samples=1):
    global  functioncode

    Fun1=functioncode[F1]
    Fun2=functioncode[F2]
    sys_prompt = (
        "A code clone refers to two or more identical or similar source code snippets existing in a code repository."
        "You are a capable software development assistant specializing in code clone detection, aiming to help other developers understand the characteristics of code clones and identify clone relationships existing in code."
    )
    prompt = (
        "Now, given the following two function snippets, please return the judgment result of code clone detection in JSON format (output in JSON Lines/JSONL format)."
        "Think step by step and provide analysis around the following aspects:"
        "Text similarity of code"
        "Semantic similarity of code"
        "Syntactic similarity of code"
        "Functional similarity of code"

        f"function snippets:\nFun1:\n{Fun1}\nFun2:\n{Fun2}" 
    )
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
        print(f"LLM fial:{str(e)}")
        return []
def process_chunk(chunk):
    F1,F2,Type=chunk

    result = call_llm(F1,F2)

    return [F1,F2,Type,result]
def init_model():
    global functioncode
    with open('dataset\Java\funs5000.json', 'r') as f:
        functioncode = json.load(f)
def main():
    init_model ()
    processed=set()
    if os.path.exists(f"{modelname}.csv"):
        print(modelname)
        with open(f"{modelname}.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                processed.add((row[0],row[1]))
    else:
        with open(f"{modelname}.csv", 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["F1","F2","type","result"])
    print(f"Already processed {len(processed)} pairs.")
    work=[]
    with open("dataset\Java\2500noclone.csv", 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row_idx, row in enumerate(reader):
            if (row[0],row[1]) in processed:
                continue
            work.append((row[0],row[1],row[2]))
    with open("dataset\Java\2500clone.csv", 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row_idx, row in enumerate(reader):
            if (row[0],row[1]) in processed:
                continue
            work.append((row[0],row[1],row[2]))
 
    q = queue.Queue()
    for k in work:
        q.put(k)

    total_tasks = q.qsize()
    print(f"Total tasks: {total_tasks}")
    with tqdm(total=total_tasks) as pbar:
        with ProcessPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(process_chunk, q.get()) for _ in range(total_tasks)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result[3]!=[]:
                        with open(f"{modelname}-1.csv", 'a', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([result[0], result[1], result[2],result[3]])
                except Exception as e:
                    print(f"Error getting result: {e}")
                pbar.update(1)
if __name__ == '__main__':
    main()
