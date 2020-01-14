import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from dataclasses import dataclass

mfcc_feature_size = 13
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
        Total score    {(stats.silence_out + stats.voice_out) / (stats.misses_s + 2 * stats.misses_v)}
    """


def plot_model_output(output, seq_length):
    plt.plot(np.linspace(0, seq_length / 100, seq_length), output)
