import requests
import os
import pickle
import json
import random
from collections import defaultdict
import numpy as np
import pandas as pd

input_data = dict()
input_data = pd.read_csv("./code/package/example/RP_1.csv", index_col=0)
#print(input_data)
input_data = input_data.to_json()
#print(input_data)
response = requests.post(
    #'http://localhost:8080/v2/models/knuh_v5/infer',
    'https://infer.nubison.io/inference-stag/knuh-v5-default/v2/models/infer', #http://localhost:8080/v2/models/knuh_v3/infer
    headers={'Content-Type': 'application/json'},
    json={
        "content_type": "pd",
        "inputs": [{
            "name": "data",
            "shape": [len(input_data)],
            "datatype": "BYTES",
            "data": [input_data],
            "parameters": {
                "content_type": "str"
            }
        }]
    })
tt = response.json()
print(tt)
