import torch
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import json
import config


trainfile_path = './Base_Dataset/KDDTest_no_headers.csv'
testfile_path = './Base_Dataset/KDDTrain_no_headers.csv'

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
             "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"]

le = LabelEncoder()

df_train = pd.read_csv(trainfile_path,header=None, names = col_names)
df_test = pd.read_csv(testfile_path, header=None, names = col_names)

#rename class
df_train = df_train.rename(columns={"class": "class_"})
df_test = df_test.rename(columns={"class": "class_"})

text_columns=['protocol_type', 'service', 'flag']

df_train_text_values = df_train[text_columns]
df_test_text_values = df_test[text_columns]

df_train_text_values_enc = df_train_text_values

#Label Encoding 
#mapping needed later in network_anomaly_detection.py
le_name_mapping = {}
for col in text_columns:
    df_train_text_values_enc[col]=le.fit_transform(df_train_text_values[col])
    le_name_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

#Saving Encoded values for later use
with open('./Data/label_encode_values.json', 'w') as data: 
      data.write('"'+str(le_name_mapping)+'"')

#df_test_text_values_enc=df_test_text_values.apply(le.fit_transform)


print('One Hot Encoding')
#One Hot Encoding
enc = OneHotEncoder(categories='auto')

#Colum Headers
protocol=sorted(df_train.protocol_type.unique())
protocol_header=['Protocol_' + x for x in protocol]

service=sorted(df_train.service.unique())
service_header=['service_' + x for x in service]

flag=sorted(df_train.flag.unique())
flag_header=['flag_' + x for x in flag]

#class_=sorted(df_train.class_.unique())
#class_header=['class_' + x for x in class_]

colum_headers_train_enc = protocol_header + service_header + flag_header #+ class_header


df_train_text_values_encenc = enc.fit_transform(df_train_text_values_enc)
df_cat_data = pd.DataFrame(df_train_text_values_encenc.toarray(),columns=colum_headers_train_enc)

#test
#df_test_text_values_encenc = enc.fit_transform(df_test_text_values_enc)
#testdf_cat_data = pd.DataFrame(df_test_text_values_encenc.toarray(),columns=colum_headers_test_enc)

list_service_train=df_train['service'].tolist()

#concat the dataframes
final_df=df_train.join(df_cat_data)
#final_df_test=df_test.join(testdf_cat_data)

#remove text columns
for chr in text_columns:
    final_df.drop(chr, axis=1, inplace=True)
 #   final_df_test.drop(chr, axis=1, inplace=True)

final_df = final_df[config.final_col_names]
final_df.to_csv('./Data/train_enc.csv', index=False)
#final_df_test.to_csv('./Data/test_enc.csv', index=False)

np.savetxt('./Data/final_col_names.txt', config.final_col_names, delimiter=',', fmt='%s')
print("Saved!")