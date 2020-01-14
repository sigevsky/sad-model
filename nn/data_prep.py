import numpy as np
import torch
import torch.utils.data as dt

from preprocess_audio import SdaContent


class SamplingDataset(dt.IterableDataset):
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self.gen


def random_gen(generators):
    while True:
        ind = np.random.randint(0, len(generators))
        yield next(generators[ind])


def generate_samples(seq_length, mfcc_features, labels):
    """
    Generates samples for sda model
    :return: Returns generator which produces tuples of size (seq_length x mfcc_feature_count, seq_length)
    where the first component is an input and second one is a label row
    """
    data_length = torch.tensor(mfcc_features.shape[0], dtype=torch.float64)
    while True:
        v = np.random.randint(0, data_length - seq_length)
        yield mfcc_features[v:v + seq_length], labels[v: v + seq_length]


def sample_interval(fcontent: SdaContent, audio_sig, rate, from_s, to_s):
    inputs = fcontent.mfcc[int(from_s / fcontent.frame_step): int(to_s / fcontent.frame_step)]
    labels = fcontent.labels[int(from_s / fcontent.frame_step): int(to_s / fcontent.frame_step)]
    sound = audio_sig[from_s * rate: to_s * rate]
    return inputs, labels, sound
