import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dt

from nn.data_prep import generate_samples, SamplingDataset
from nn.sda_nn import SdaNet
from preprocess_audio import SdaContent


def load_data(mc_file) -> SdaContent:
    with open(mc_file, 'rb') as f:
        return pickle.load(f)


batch_size = 100
mfcc_feature_size = 13
hidden_size = 20
seq_length = 1500
ep_step = 100

net = SdaNet(mfcc_feature_size, hidden_size).double()

criterion = nn.BCELoss().double()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

data = load_data("../data/res/young_pope/typ1.mc")
dl = dt.DataLoader(SamplingDataset(generate_samples(seq_length, data.mfcc, data.labels)),
                   batch_size=batch_size, drop_last=False, shuffle=False)

running_loss = 0.0
for epoch in range(0, 20):
    if running_loss < 1. and running_loss != .0:
        break
    if epoch % 2 == 1:  # print every 2 epochs
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, (epoch + 1) * ep_step, running_loss / (5 * ep_step)))
        running_loss = 0.0
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

torch.save(net, f"../data/res/models/model-{time.time()}.dict")
print("Done!")
