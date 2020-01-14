from datetime import timedelta
from typing import List, Tuple

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
    Prepares mfcc features from a given wav file.
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
    return not (c.startswith("<font") or c.startswith("<b>") or ("subtitle" in c) or ("www" in c))


def compensate_lag(lag: timedelta):
    """
    Returns wrapper that will adjust start and end time of each subtitle by some lag value.
    :param lag: Lag value
    :return: Subtitle -> Subtitle function
    """

    def go(sub: srt.Subtitle) -> srt.Subtitle:
        return srt.Subtitle(sub.index, sub.start + lag, end=sub.end + lag, content=sub.content,
                            proprietary=sub.proprietary)

    return go


def shrink_interval(lag: timedelta):
    """
    Adds lag only to the end of the interval.
    :param lag: Lag value
    :return: Subtitle -> Subtitle function
    """

    def go(sub: srt.Subtitle) -> srt.Subtitle:
        return srt.Subtitle(sub.index, sub.start, end=sub.end + lag, content=sub.content,
                            proprietary=sub.proprietary)

    return go


def merge_close_subtitles(subs: List[srt.Subtitle], dist=0.05) -> List[srt.Subtitle]:
    """
    Usually one long phrase is broken down to multitude of subtitles having
    some fixed time distance between them. This method reconstruct it back from separate sub pieces.
    :param subs: List of subtitles
    :param dist: Distance between two subtitles in order to be considered as part of one
    :return: List of subtitles where subtitles having overlapping
     with regard to dist intervals are merged into one
    """
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


def intervals_from_subtitles(subs: List[srt.Subtitle],
                             shrink: timedelta = timedelta(seconds=0),
                             lag: timedelta = timedelta(seconds=0)):
    return filter(is_not_misc_sub,
                  map(compensate_lag(lag),
                      map(shrink_interval(shrink),
                          merge_close_subtitles(subs))))


def extract_voice_intervals(audio_file: str,
                            sub_file: str,
                            shrink: timedelta = timedelta(seconds=0),
                            lag: timedelta = timedelta(seconds=0)) -> Tuple[int, List[np.ndarray]]:
    """
    Useful for testing accuracy of subtitles in tandem with wavfile.write("file_name.wav", rate, intr[numb]).
    :param shrink: Lag added to the end time of subtitle in order to shrink the interval
    :param audio_file: Path of a file from which voice intervals should be extracted
    :param sub_file: Path of a file with subtitles corresponding to the audio file
    :param lag: Lag value
    :return: Tuple
        0: Rate of the audio which has been processed
        1: List of intervals containing voice samples
    """
    rate, sig = wavfile.read(audio_file)

    def to_sample_intervals(sub: srt.Subtitle):
        def pos(date: timedelta):
            return int(round(date.total_seconds() * rate))

        return sig[pos(sub.start): pos(sub.end)]

    with open(sub_file, 'r', encoding='ISO-8859-15') as subs_raw:
        subs = list(srt.parse(subs_raw.read()))
        return rate, list(map(to_sample_intervals, intervals_from_subtitles(subs, shrink=shrink, lag=lag)))


def prepare_labels_from_subs(subs: List[srt.Subtitle],
                             frame_step,
                             shrink: timedelta = timedelta(seconds=0),
                             lag: timedelta = timedelta(seconds=0)):
    """
    Prepares labels which will be supplied to the model.
    :param shrink: Lag added to the end time of subtitle in order to shrink the interval
    :param subs: List of subtitles
    :param frame_step: Frame step that is used on mfcc feature generation (in seconds)
    :param lag: Lag value
    :return: List of labels having value either 0 or 1,
     where zero stands for absence of the voice in the given interval and one for presence.
     The values are sampled with frame_step interval
    """

    def pos(date: timedelta):
        return int(round(date.total_seconds() / frame_step))

    def to_sample_intervals(sub: srt.Subtitle):
        return pos(sub.start), pos(sub.end)

    intervals = list(map(to_sample_intervals, intervals_from_subtitles(subs, shrink=shrink, lag=lag)))

    labels = np.zeros(intervals[-1][1])  # end of the last interval
    for a, b in intervals:
        labels[a: b] = np.ones(b - a)

    return labels


def prepare_test_data(audio_file, subs_file,
                      output_file,
                      shrink=timedelta(seconds=0.0),
                      lag=timedelta(seconds=0.1)):
    """
    Extracts mfcc features from audio file along with with labels marking each sample with either 0 or 1 value
    corresponding to absence or presence of the voice is sampled piece accordingly.
    :param shrink: Lag added to the end time of subtitle in order to shrink the interval
    :param audio_file: The path to audio file
    :param subs_file: The path to sub file
    :param output_file: The path where SdaContent should be saved to
    :param lag: Lag of the file
    :return:
    """
    frame_step = 0.01
    frame_size = 0.025
    with open(subs_file, 'r', encoding='ISO-8859-15') as subs_raw:
        subs = list(srt.parse(subs_raw.read()))
        labels = prepare_labels_from_subs(subs, frame_step, shrink=shrink, lag=lag)
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
