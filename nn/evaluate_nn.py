import pickle

import numpy.ma as ma
import torch

from nn.data_prep import generate_samples
from nn.sda_nn import SdaNet
from preprocess_audio import SdaContent


def load_data(mc_file) -> SdaContent:
    with open(mc_file, 'rb') as f:
        return pickle.load(f)


mfcc_feature_size = 13
seq_length = 2000
err_threshold = .5


def evaluate_model(net: SdaNet):
    eval_data = load_data("../data/res/young_pope/typ1.mc")
    inputs, labels = next(generate_samples(seq_length, eval_data.mfcc, eval_data.labels))
    # output has a shape of 1 x 2000 x 1 so we need to get rid of 1th dimensions
    output = net(torch.tensor(inputs.reshape(1, seq_length, -1), dtype=torch.float64)).detach().numpy()[0, :, 0]
    silence_out = ma.array(output, mask=list(map(lambda e: True if e == 1 else False, labels)))
    voice_out = ma.array(output, mask=list(map(lambda e: True if e == 0 else False, labels)))
    print(f"Silent samples {silence_out.count()}, voice samples: {voice_out.count()}")
    # labels are zero on voiceless intervals
    hits_s = ma.where(silence_out < err_threshold)[0].size
    misses_s = ma.where(silence_out > err_threshold)[0].size
    score_s = hits_s / misses_s
    print(f"Silence hits {hits_s}, misses {misses_s}, score {score_s}")
    # labels are exactly ones on voice intervals
    hits_v = ma.where(voice_out > err_threshold)[0].size
    misses_v = ma.where(voice_out < err_threshold)[0].size
    score_v = hits_v / misses_v
    print(f"Voiced hits {hits_v}, misses {misses_v}, score {score_v}")
    print(f"Total score {seq_length / (misses_s + 2 * misses_v)}")


evaluate_model(torch.load("../data/res/models/model-1578512869.616221.dict"))

