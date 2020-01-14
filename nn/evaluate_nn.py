import pickle
from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
import torch
import matplotlib.pyplot as plt

from nn.data_prep import generate_samples
from preprocess_audio import SdaContent


def load_data(mc_file) -> SdaContent:
    with open(mc_file, 'rb') as f:
        return pickle.load(f)


# N = 7
# colors = [hsv_to_rgb((x * 1.0 / N, 0.5, 0.5)) for x in range(N)]

mfcc_feature_size = 13
seq_length = 3000
err_threshold = .5


@dataclass
class EvalStats:
    silence_out: int
    voice_out: int
    hits_s: int
    misses_s: int
    score_s: float
    hits_v: int
    misses_v: int
    score_v: float


def get_eval_stats(outputs, labels):
    silence_out = ma.array(outputs, mask=list(map(lambda e: True if e == 1 else False, labels)))
    voice_out = ma.array(outputs, mask=list(map(lambda e: True if e == 0 else False, labels)))
    # labels are 0 on voiceless intervals
    hits_s = ma.where(silence_out < err_threshold)[0].size
    misses_s = ma.where(silence_out > err_threshold)[0].size
    score_s = hits_s / misses_s
    # labels are exactly 1 on voice intervals
    hits_v = ma.where(voice_out > err_threshold)[0].size
    misses_v = ma.where(voice_out < err_threshold)[0].size
    score_v = hits_v / misses_v
    return EvalStats(silence_out.count(), voice_out.count(), hits_s, misses_s, score_s, hits_v, misses_v, score_v)


def format_model_stats(stats: EvalStats):
    return f"""
        Silent samples {stats.silence_out}, voice samples: {stats.voice_out}
        Silence hits   {stats.hits_s}     , misses {stats.misses_s}, score {stats.score_s}
        Voiced hits    {stats.hits_v}     , misses {stats.misses_v}, score {stats.score_v}
        Total score    {seq_length / (stats.misses_s + 2 * stats.misses_v)}
    """


def plot_model_output(output):
    plt.plot(np.linspace(0, seq_length / 100, seq_length), output)


def generate_eval_sample(source):
    eval_data = load_data(source)
    return next(generate_samples(seq_length, eval_data.mfcc, eval_data.labels))


# TODO: move this one to notebook
def evaluate_models():
    nets = [torch.load(f"../data/res/models/1578999854657/model-{i}.0.dict") for i in [1, 7]]
    for net in nets:
        net.eval()
    inputs, labels = generate_eval_sample('../data/res/young_pope/typ8.mc')
    # output has a shape of 1 x 2000 x 1 so we need to get rid of 1th dimensions
    outs = [net(torch.tensor(inputs.reshape(1, seq_length, -1), dtype=torch.float64)).detach().numpy()[0, :, 0]
            for net in nets]
    stats = [get_eval_stats(output, labels) for output in outs]
    for stat in stats:
        print(format_model_stats(stat))

    # plot models predictions
    plt.plot(np.linspace(0, seq_length / 100, seq_length), labels)
    for out in outs:
        plot_model_output(out)

    plt.show()


evaluate_models()

# plt.plot(np.linspace(0, seq_length / 100, seq_length), output)
# plt.plot(np.linspace(0, seq_length / 100, seq_length), labels)
# plt.show()
# # output has a shape of 1 x 2000 x 1 so we need to get rid of 1th dimensions
# output = net(torch.tensor(inputs.reshape(1, seq_length, -1), dtype=torch.float64)).detach().numpy()[0, :, 0]