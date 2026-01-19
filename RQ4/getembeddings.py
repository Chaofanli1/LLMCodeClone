import torch
import json
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re

# Instructions for use
# Install the dependency
# MODEL_PATH: modify the actual storage location of your model
# INPUT_FILE: modify to your function code data file path, json format ("idx": "code")
# OUTPUT_FILE: modify the output file name as needed
MODEL_PATH = "/home/data/codebert"
INPUT_FILE = "/home/data/embeddings_codebert/data.json" 
OUTPUT_FILE = "/home/data/embeddings_codebert/embeddings_output2.jsonl" 

def main():

    print(f"Loading the model: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def get_embedding(code_str):
        code_str = re.sub(r'package\s+[\w\.]+;', '', code_str)
        code_str = re.sub(r'import\s+[\w\.\*]+;', '', code_str)
        code_str = re.sub(r'\s+', ' ', code_str).strip()

        inputs = tokenizer(code_str, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model(**inputs)[0]
            embedding = output[0, 0, :].cpu().numpy().tolist()
            
        return embedding

    print(f"Reading input: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    print(f"Start processing： {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for func_id, func_code in tqdm(data_dict.items(), desc="Generating"):
            if func_code:
                try:
                    vec = get_embedding(func_code)
                    record = {
                        "idx": func_id,
                        "embedding": vec
                    }
                    fout.write(json.dumps(record) + "\n")
                    
                except Exception as e:
                    print(f"Error {func_id}: {e}")

if __name__ == "__main__":
    main()