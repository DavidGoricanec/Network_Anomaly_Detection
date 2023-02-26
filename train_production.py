import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
import copy
import numpy as np
from classes.Deep import Deep

#train pytorch

trainfile_path = './Data/train_enc.csv'
model_path = './Data/model.pth'

col_names = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
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

col_length = len(col_names)-1
threshold = 0.2

data = pd.read_csv(trainfile_path, header=1)
x = data.iloc[:, 0:col_length]
y = data.iloc[:, col_length]
 
#In convert_dataset.py we only label-encoded the data that also got one hot encoded. This excluded the "class" column 
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

#anomaly = 0; normal = 1

x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model = Deep()

# train one model
def model_train(model, X_train, y_train, X_val, y_val):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
    n_epochs = 200
    batch_size = 10
    batch_start = torch.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc
 
# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.99, shuffle=True)

print("Starting training")
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Training over! Final model accuracy: {acc*100:.2f}%")

torch.save(model.state_dict(), model_path)
print("Pytorch Model Saved!")