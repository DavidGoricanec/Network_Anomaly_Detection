import torch
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import json


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
le_name_mapping = {}
for col in text_columns:
    df_train_text_values_enc[col]=le.fit_transform(df_train_text_values[col])
    le_name_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

with open('./Data/label_encode_values.txt', 'w') as data: 
      data.write(str(le_name_mapping))

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

#service_test_header=sorted(df_test.service.unique())
#service_header_test =['service_' + x for x in service_test_header]

#colum_headers_test_enc = protocol_header + service_header_test + flag_header #+ class_header


df_train_text_values_encenc = enc.fit_transform(df_train_text_values_enc)
df_cat_data = pd.DataFrame(df_train_text_values_encenc.toarray(),columns=colum_headers_train_enc)

#test
#df_test_text_values_encenc = enc.fit_transform(df_test_text_values_enc)
#testdf_cat_data = pd.DataFrame(df_test_text_values_encenc.toarray(),columns=colum_headers_test_enc)

#remove differences
list_service_train=df_train['service'].tolist()
#list_service_test= df_test['service'].tolist()
#diff=list(set(list_service_train) - set(list_service_test))
#for col in ['service_' + x for x in diff]:
 #   testdf_cat_data[col] = 0

#concat the dataframes
final_df=df_train.join(df_cat_data)
#final_df_test=df_test.join(testdf_cat_data)

#remove text columns
for chr in text_columns:
    final_df.drop(chr, axis=1, inplace=True)
 #   final_df_test.drop(chr, axis=1, inplace=True)

final_col_names = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
             "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "Protocol_icmp","Protocol_tcp",
             "Protocol_udp","service_IRC","service_X11","service_Z39_50","service_auth","service_bgp","service_courier","service_csnet_ns",
             "service_ctf","service_daytime","service_discard","service_domain","service_domain_u","service_echo","service_eco_i","service_ecr_i",
             "service_efs","service_exec","service_finger","service_ftp","service_ftp_data","service_gopher","service_hostnames","service_http",
             "service_http_443","service_imap4","service_iso_tsap","service_klogin","service_kshell","service_ldap","service_link","service_login",
             "service_mtp","service_name","service_netbios_dgm","service_netbios_ns","service_netbios_ssn","service_netstat","service_nnsp","service_nntp",
             "service_ntp_u","service_other","service_pm_dump","service_pop_2","service_pop_3","service_printer","service_private","service_remote_job","service_rje",
             "service_shell","service_smtp","service_sql_net","service_ssh","service_sunrpc","service_supdup","service_systat","service_telnet","service_tftp_u",
             "service_tim_i","service_time","service_urp_i","service_uucp","service_uucp_path","service_vmnet","service_whois","flag_OTH","flag_REJ","flag_RSTO",
             "flag_RSTOS0","flag_RSTR","flag_S0","flag_S1","flag_S2","flag_S3","flag_SF","flag_SH","class_"]

final_df = final_df[final_col_names]
final_df.to_csv('./Data/train_enc.csv', index=False)
#final_df_test.to_csv('./Data/test_enc.csv', index=False)

np.savetxt('final_col_names.txt', final_col_names, delimiter=',', fmt='%s')