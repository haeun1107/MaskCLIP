import json, os

with open('data/BTCV/dataset.json') as f:
    data = json.load(f)

os.makedirs('data/BTCV', exist_ok=True)

with open('data/BTCV/train.txt', 'w') as f:
    for entry in data['training']:
        base = os.path.basename(entry['image']).replace('.png', '')
        f.write(base + '\n')

import json
import os

# BTCV dataset.json 열기
with open('data/BTCV/dataset.json') as f:
    data = json.load(f)

# val.txt 생성
with open('data/BTCV/val.txt', 'w') as f:
    for entry in data['test']:
        base = os.path.basename(entry['image']).replace('.png', '')
        f.write(base + '\n')
