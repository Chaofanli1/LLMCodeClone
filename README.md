# LLMCodeClone
Prompt-Driven Large Language Models for Automated Code Clone Detection: An Empirical Study

## Dataset
This repository releases the Java code clone dataset and cross-language code clone dataset constructed in the paper, stored in the `dataset/` folder.  
- Code clone pairs (positive/negative) are stored in **CSV files** (with annotations like clone type and similarity scores).  
- Corresponding function source code is stored in **JSON files**, indexed by unique IDs to match CSV entries.


## Model Inference
All invocations of open-source models in this paper use the batch inference function provided by [LlamaFactory](https://github.com/hiyouga/LlamaFactory/tree/main).

## Research Resources
- Scripts for calling commercial models in RQ1, RQ2, and RQ5 are stored in their respective directories (`RQ1/`, `RQ2/`, `RQ5/`).
- In `RQ2/`, we provide code clone examples identified by LLMs that cannot be detected by traditional tools.
- In `RQ4/`, we release the code for generating embedding vectors and the corresponding embedding files.