import torch
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


trainfile_path = './Base_Dataset/KDDTest_no_headers.csv'
testfile_path = './Base_Dataset/KDDTrain_no_headers.csv'

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
             "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"]

df_train = pd.read_csv(trainfile_path,header=None, names = col_names)
df_test = pd.read_csv(testfile_path, header=None, names = col_names)


text_columns=['protocol_type', 'service', 'flag','class']

df_train_text_values = df_train[text_columns]
df_test_text_values = df_test[text_columns]

df_train_text_values_enc=df_train_text_values.apply(LabelEncoder().fit_transform)
df_test_text_values_enc=df_test_text_values.apply(LabelEncoder().fit_transform)

print('Train Conversion')
print('-------------------------')
print(df_train_text_values.head())
print('-------------------------')
print(df_train_text_values_enc.head())
print('-------------------------')

print('Test Conversion')
print('-------------------------')
print(df_test_text_values.head())
print('-------------------------')
print(df_test_text_values_enc.head())
print('-------------------------')


#print(type(df_test_text_values_enc))

df_train_text_values_enc.to_csv('train_enc.csv', index=False)
df_test_text_values_enc.to_csv('test_enc.csv', index=False)