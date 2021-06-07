import json
import pdb
from pathlib import Path
import numpy as np

models = ['candle', 'resnet', 'vgg']

instances = ['c5.4xlarge', 'c5a.4xlarge', 'g4dn.2xlarge', 'm5n.2xlarge', 'm5.2xlarge', 'r5.xlarge', 'r5n.xlarge', 't3.2xlarge']

for ins in instances:
    # make directory if does not exist 
    Path(f'/home/baolin/GIT/SIMBO/characterization/logs/{ins}').mkdir(parents=True,exist_ok=True)
    for model in models:
        for batch in range(10, 501, 10):
            batch = str(batch)
            path = f'../{ins}/{model}_{batch}_1.json'
            with open(path, 'r') as f:
                lat_list = json.load(f)

            with open(f'/home/baolin/GIT/SIMBO/characterization/logs/{ins}/{model}_{batch}_1.json', 'w') as f:
                json.dump(lat_list, f, indent=4)



