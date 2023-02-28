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
import json
import config
import requests

encoded_values = np.loadtxt('./Data/label_encode_values.txt', delimiter=',', dtype=str)

with open('./Data/label_encode_values.json', 'r') as file:
    # Load the JSON data from the file
    # Check why it only works this way.
    str = str(file.read())
    str = str.replace('"', '')
    str = str.replace("'", '"')
    json_data = json.loads(str)

def my_print(str):
    print('-----------------------------------------------')
    print(str)
    print('-----------------------------------------------')

def default():
    data = {col: [0] for col in config.col_names}
    return data
    #return pd.DataFrame(data)

def set_packet_data(packet, data):
    #Protocol
    proto = ''
    print(packet[IP])
    if packet.haslayer(TCP):
        proto = 'TCP'
    elif packet.haslayer(UDP):
        proto = 'UDP'
    else: #Model was trained for TCP, UDP and ICMP
        proto = 'ICMP'

    proto_enc = json_data['protocol_type'][proto.lower()]
    data['Protocol_' + proto.lower()] = proto_enc
    
    #Service
    try:
        service_port = packet[proto].sport
        service_name = config.ports_mapping[service_port]
        data['service_' + service_name] = json_data['service'][service_name]
    except: 
        print("Could not find service")
    
    #flags
    #for simplicity we set the connection to established
    data['flag_S1'] = json_data['flag']['S1']
    data['src_byte'] = packet[IP].len
    my_print(packet[IP].flags)
    #print(data)
    return data

def handle_packet(packet, my_model):
    packet.show()
    packet.show2()
    if packet.haslayer(IP):
        data_arr = default()
        data_arr = set_packet_data(packet, data_arr)
        data = pd.DataFrame(data_arr)
        x_test = data.iloc[:, 0:config.col_length]
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_pred = model(x_test)
        y_pred = (y_pred > config.threshold).float() # 0.0 or 1.0
        if y_pred == 1:
            #normal
            print("normal packet")
        else:
            print("anomaly detected!")
            #send to home server for anomaly alert 
            #sending the packet as a parameter 
            requests.post(config.url,data_arr, timeout=3) #comment out when raspi is offline

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
        #y_pred = (y_pred > config.threshold).float() # 0.0 or 1.0
        #print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

