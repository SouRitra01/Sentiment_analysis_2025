import numpy as np
print(np.__version__)
print(np.array([1, 2, 3]))

import torch
print(torch.__version__)
print(torch.tensor([1,2,3]))


from transformers import TrainingArguments
print("transformers is working!")

args = TrainingArguments(
    output_dir='./output',
    evaluation_strategy='epoch'
)
print("SUCCESS! Everything works.")
