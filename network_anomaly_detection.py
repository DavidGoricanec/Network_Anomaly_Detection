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

def handle_packet(packet, my_model):
    # Process the packet payload using my_param here
    print(dir(packet))

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

