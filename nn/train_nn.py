import pickle
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dt

from nn.data_prep import generate_samples, SamplingDataset, random_gen
from nn.sda_nn import SdaNet
from preprocess_audio import SdaContent

run_time = millis = int(round(time.time() * 1000))
os.makedirs(f"../data/res/models/{run_time}/")


def load_data(mc_file) -> SdaContent:
    with open(mc_file, 'rb') as f:
        return pickle.load(f)


def checkpoint(i, cnet):
    torch.save(cnet, f"../data/res/models/{run_time}/model-{i}.dict")


batch_size = 100
mfcc_feature_size = 13
hidden_size = 20
seq_length = 1500
ep_step = 100

net = SdaNet(mfcc_feature_size, hidden_size).double()

criterion = nn.BCELoss().double()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

data = [load_data(f"../data/res/young_pope/typ{i}.mc") for i in range(1, 7)]
data_gen = random_gen([generate_samples(seq_length, dt.mfcc, dt.labels) for dt in data])
dl = dt.DataLoader(SamplingDataset(data_gen),
                   batch_size=batch_size, drop_last=False, shuffle=False)

running_loss = 0.0
for epoch in range(0, 200):
    if running_loss < 1. and running_loss != .0:
        break
    if epoch % 2 == 1:  # print every 2 epochs
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, (epoch + 1) * ep_step, running_loss / (5 * ep_step)))
        running_loss = 0.0
    if epoch % 10 == 9:
        checkpoint((epoch + 1) / 10, net)

    for j, (inputs, labels) in enumerate(dl):
        if j == ep_step:
            break
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.view(batch_size, seq_length), labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

checkpoint(0, net)
print("Done!")
