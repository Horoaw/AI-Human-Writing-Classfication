import random
from datasets import load_dataset

dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")

def clean_text(text):
    return text.strip()

human_samples = []
generated_samples = []

for example in dataset:
    text = clean_text(example['text'])
    source = example['source']
    
    if not text:
        continue
    
    sample = {'text': text}
    
    if source == 'human':
        human_samples.append(sample)
    elif source == 'ai':
        generated_samples.append(sample)

print(f"There will be {len(human_samples)} writen human，and {len(generated_samples)}  generated。")

# number that u wanna use
num_samples = 10000

human_samples = random.sample(human_samples, num_samples)
generated_samples = random.sample(generated_samples, num_samples)

# Lable
for s in human_samples:
    s['label'] = 0
for s in generated_samples:
    s['label'] = 1


balanced_data = human_samples + generated_samples
random.shuffle(balanced_data)

print(f"The dataset will be : {len(balanced_data)}")

#  pandas DataFrame
import pandas as pd

df = pd.DataFrame(balanced_data)
df.to_csv('balanced_100k_dataset.csv', index=False)
