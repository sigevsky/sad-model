from datetime import timedelta
from typing import List

import python_speech_features as psf
import numpy as np
import srt
from scipy.io import wavfile
import pickle


class SdaContent:
    def __init__(self, rate, frame_step, content, labels):
        self.rate = rate
        self.frame_step = frame_step
        self.mfcc = content
        self.labels = labels


def split_by_chunks(data, n):
    return [data[i:i + n] for i in range(0, data.shape[0], n)]


# TODO: eliminate lag added after each chunk, now it equates to
#  abs(frame_step (0.01 * rate) - sig.shape[0] / np.concatenate(mfcc_data).shape[0])
#  error: (sig.shape[0] / (0.01 * rate) - np.concatenate(mfcc_data).shape[0]) / rate | per second
def prepare_mfcc(sig, rate, frame_step, frame_size):
    """
    Prepares mfcc features from a given wav file
    :param sig: audion signal
    :param rate: rate of the audio signal
    :param frame_step: frame step in seconds
    :param frame_size: frame size in seconds
    :return: 13 mfcc features in each row corresponding to single frame step
    """
    chunked_sig = split_by_chunks(sig, rate * 60 * 20)  # twenty minutes chunks

    mfcc_data = [psf.mfcc(chunk,
                          samplerate=rate,
                          winfunc=np.hamming,
                          nfft=int(frame_size * rate),
                          winstep=frame_step)
                 for chunk in chunked_sig]
    return np.concatenate(mfcc_data)


def is_not_misc_sub(sub: srt.Subtitle):
    c = sub.content
    return not (c.startswith("<font") or c.startswith("<b>"))


def compensate_lag(lag: timedelta):
    def go(sub: srt.Subtitle):
        return srt.Subtitle(sub.index, sub.start + lag, end=sub.end + lag, content=sub.content,
                            proprietary=sub.proprietary)
    return go


def merge_close_subtitles(subs: List[srt.Subtitle], dist=0.05):
    starts = np.array(list(map(lambda sub: sub.start.total_seconds(), subs)))[1:]
    ends = np.array(list(map(lambda sub: sub.end.total_seconds(), subs)))[:-1]
    to_merge = (np.round(starts - ends)) <= dist
    merged_subs = [subs[0]]
    for i, p in enumerate(to_merge, 1):
        if p:
            f = merged_subs[-1]
            s = subs[i]
            merged_subs[-1] = srt.Subtitle(f.index, f.start, end=s.end, content=f"{f.content}\n{s.content}",
                                           proprietary=f.proprietary)
        else:
            sub = subs[i]
            nsub = srt.Subtitle(len(merged_subs) + 1, sub.start, end=sub.end, content=sub.content,
                                proprietary=sub.proprietary)
            merged_subs.append(nsub)
    return merged_subs


def extract_voice_intervals(audio_file: str, sub_file: str, lag: timedelta = timedelta(seconds=0)):
    rate, sig = wavfile.read(audio_file)

    def to_sample_intervals(sub: srt.Subtitle):
        def pos(date: timedelta):
            return int(round(date.total_seconds() * rate))
        return sig[pos(sub.start): pos(sub.end)]

    with open(sub_file, 'r', encoding='ISO-8859-15') as subs_raw:
        subs = list(srt.parse(subs_raw.read()))
        return rate, list(map(to_sample_intervals,
                              filter(is_not_misc_sub,
                                     map(compensate_lag(lag),
                                         merge_close_subtitles(subs)))))


def prepare_labels_from_subs(subs: List[srt.Subtitle], frame_step, lag: timedelta = timedelta(seconds=0)):
    def pos(date: timedelta):
        return int(round(date.total_seconds() / frame_step))

    def to_sample_intervals(sub: srt.Subtitle):
        return pos(sub.start), pos(sub.end)

    intervals = list(map(to_sample_intervals,
                         filter(is_not_misc_sub,
                                map(compensate_lag(lag),
                                    merge_close_subtitles(subs)))))

    labels = np.zeros(intervals[-1][1])  # end of the last interval
    for a, b in intervals:
        labels[a: b] = np.ones(b - a)

    return labels


def prepare_test_data(audio_file, subs_file, output_file):
    frame_step = 0.01
    frame_size = 0.025
    with open(subs_file, 'r', encoding='ISO-8859-15') as subs_raw:
        subs = list(srt.parse(subs_raw.read()))
        labels = prepare_labels_from_subs(subs, frame_step, lag=timedelta(seconds=0.35))
    rate, sig = wavfile.read(audio_file)
    mfcc_features = prepare_mfcc(sig, rate, frame_step, frame_size)
    labels = np.pad(labels, (0, mfcc_features.shape[0] - labels.shape[0]), constant_values=.0)
    test_data = SdaContent(rate, frame_step, mfcc_features, labels)
    with open(output_file, 'wb') as output:
        pickle.dump(test_data, output)

# prepare_test_data("data/raw/young_pope/audio/typ1.wav",
#                   "data/raw/young_pope/subtitles/The Young Pope - 1x01 - Episode 1.HDTV.FLEET.en.srt",
#                   "data/res/young_pope/typ1.mc")

# itr = prepare_labels_from_subs("data/raw/young_pope/subtitles/The Young Pope - 1x01 - Episode 1.HDTV.FLEET.en.srt",
#                                0.01)
# print(itr[57])
# print(len(itr))
# print(prepare_mfcc("data/raw/young_pope/audio/typ1.wav"))
# rate, intr = extract_voice_intervals("data/raw/young_pope/audio/typ1.wav",
#                                      "data/raw/young_pope/subtitles/The Young Pope - 1x01 - Episode 1.HDTV.FLEET.en.srt",
#                                      timedelta(seconds=0.35))
# wavfile.write("some.wav", rate, intr[57])
# print(len(intr))
