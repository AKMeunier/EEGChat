import os
import json
import numpy as np
from datetime import datetime
from psychopy import parallel
from util.eeg.online import CVEPContinuousDecoder


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def datestr():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def save_json(dictionary, file_name):
    def default(obj):
        if isinstance(obj, (np.ndarray, np.int64, np.int32, np.float64, np.float32)):
            return obj.tolist()
        if isinstance(obj, CVEPContinuousDecoder):
            return "CVEPContinuousDecoderCNN"
        if isinstance(obj, parallel.ParallelPort):
            return obj.port._name
        if obj is None:
            return "None"
        raise TypeError('Not serializable. Define default treatment of objects of type {}'.format(type(obj)))

    with open(file_name, 'w') as fl:
        json.dump(dictionary, fl, default=default)


def load_json(file_name):
    with open(file_name, 'r') as fl:
        d = json.load(fl)
    return d


def possible_freqs(monitor_freq, dist_max=None, fmin=None, fmax=None):
    """
    Compute the frequencies that are possible to display with a monitor with the given refresh rate.
    :param monitor_freq: float; refresh rate of the monitor
    :param dist_max: int; maximum distance between to blinks in frames to consider. If None, dist_max = monitor_freq.
    :param fmin: float; Minimum frequency tobe returned
    :param fmax: float; Maximum frequency to be returned
    :return: (np.ndarray, np.ndarray); frequencies, frame_distances
    """
    if dist_max is None:
        dist_max = monitor_freq
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = monitor_freq / 2

    distances = np.arange(2, dist_max + 1)
    freqs = monitor_freq / distances

    mask = (freqs >= fmin) & (freqs <= fmax)

    return freqs[mask], distances[mask].astype(int)

