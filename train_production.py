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
import config

#train pytorch

trainfile_path = config.trainfile_path
model_path = config.testfile_path

data = pd.read_csv(trainfile_path, header=1)
x = data.iloc[:, 0:config.col_length]
y = data.iloc[:, config.col_length]
 
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