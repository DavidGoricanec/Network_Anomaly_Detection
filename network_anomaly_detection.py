import functools
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import tqdm
import copy
import numpy as np
from classes.Deep import Deep
from scapy.all import *

col_names = np.loadtxt('final_col_names.txt', delimiter=',', dtype=str)
col_length = len(col_names)-1
threshold = 0.2

def default_df():
    data = {col: [0] for col in col_names}
    return pd.DataFrame(data)

def handle_packet(packet, my_model):
    # Process the packet payload using my_param here
    print(dir(packet))
    data = default_df()
    x_test = data.iloc[:, 0:col_length]
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_pred = model(x_test)
    y_pred = (y_pred > threshold).float() # 0.0 or 1.0
    print(y_pred)

model_path = './Data/model.pth'
model = Deep()
model.load_state_dict(torch.load(model_path))

partial_handle_packet = functools.partial(handle_packet, my_model=model)

model.eval()
with torch.no_grad():
    print('Start scanning network traffic')
    sniff(prn=partial_handle_packet, store=0)
    # Test out inference with 5 samples
    #for i in range(5):
        #y_pred = model(X_test[i:i+1])
        #y_pred = (y_pred > threshold).float() # 0.0 or 1.0
        #print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

