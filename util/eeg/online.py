import abc
from abc import ABC
import numpy as np
import pickle
from scipy import signal
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import time
import warnings
import keras

from util.eeg.models import cvep_cnn_nagel2019


def start_lsl_stream():
    """
    Starts listening to EEG lsl stream. Will get "stuck" if no stream is found.
    :return: lsl_inlet; pysls.StreamInlet object
    """
    streams = resolve_stream('type', 'EEG')
    if len(streams) > 1:
        warnings.warn('Number of EEG streams is > 0, picking the first one.')
    lsl_inlet = StreamInlet(streams[0])
    lsl_inlet.pull_sample()  # need to pull first sample to get buffer started for some reason
    print("Stream started.")
    return lsl_inlet


# def start_decoder_lsl_stream_outlet():
#     info = StreamInfo(name="decoder", type="custom", channel_count=1, source_id="decoder1234")
#     outlet = StreamOutlet(info)
#     print("Decoder stream started.")
#     return outlet
#
#
# def start_decoder_lsl_stream_inlet():
#     """
#     Starts listening to the lsl stream of the decoder. Will get "stuck" if no stream is found.
#     :return: lsl_inlet; pysls.StreamInlet object
#     """
#     streams = resolve_stream('name', 'decoder')
#     if len(streams) < 1:
#         raise ConnectionError("No decoder lsl stream found. Have you started the decoder script?")
#     if len(streams) > 1:
#         warnings.warn('Number of decoder streams is > 0, picking the first one.')
#     lsl_inlet = StreamInlet(streams[0])
#     lsl_inlet.pull_sample(timeout=0.01)  # need to pull first sample to get buffer started for some reason
#     print("Decoder stream started.")
#     return lsl_inlet


def lsl_trigger_decode(trigger):
    """
    Convert lsl trigger stream triggers to the trigger sent from neurone (int between 0 and 255).
    :param trigger: int or float or np.array; array of the lsl trigger channel sent by neurrone (or signle int / float trigger)
    :return: int
    """
    if type(trigger) == np.ndarray:
        return (trigger / 256).astype(int)
    elif type(trigger) == list:
        return [int(t/256) for t in trigger]
    else:
        return int(trigger / 256)


def lsl_trigger_encode(trigger):
    """
    Convert the trigger sent from neurone (int between 0 and 255) to lsl trigger stream triggers.
    :param trigger: int or float or np.array; array of the lsl trigger channel sent by neurrone (or signle int / float trigger)
    :return: int
    """
    if type(trigger) == np.ndarray:
        return (trigger * 256).astype(int)
    else:
        return int(trigger * 256)


def pull_from_buffer(lsl_inlet, max_tries=10):
    """
    Pull data from the provided lsl inlet and return it as an array.
    :param lsl_inlet: lsl inlet object
    :param max_tries: int; number of empty chunks after which an error is thrown.
    :return: np.ndarray of shape (n_samples, n_channels)
    """
    # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
    if lsl_inlet is None:
        return

    pull_at_once = 10000
    samps_pulled = 10000
    n_tries = 0

    samples = []
    while samps_pulled == pull_at_once:
        data, _ = lsl_inlet.pull_chunk(max_samples=pull_at_once)
        arr = np.array(data)
        if len(arr) > 0:
            samples.append(arr)
            samps_pulled = len(arr)
        else:
            n_tries += 1
            time.sleep(0.01)
            if n_tries == max_tries:
                raise ValueError("Stream does not seem to provide any data.")
    return np.vstack(samples)


def cut_epochs(data_array, trigger, trigger_channel=-1, remove_trigger_ch=False, cut_to_length=None, min_samples=10):
    """
    Cuts data array obtained from lsl stream into one or several epochs according to specified triggers.
    :param data_array: np.ndarray of shape (n_samples, n_channels)
    :param trigger: int or (int, int); if a tuple (t1, t2), will cut epochs starting at trigger t1 and ending at
    trigger t2. If int t, will split into epochs at triggers t
    :param trigger_channel: int; index of the trigger channel
    :param remove_trigger_ch: bool; whether to remove the trigger channel
    :param cut_to_length: str or int;  'min': cut to the length of the shortest epoch, 'max': cut to the length of the
    longest epoch, int: cut to this number of samples, None do not cut (and return as list)
    :param min_samples: int< minimum numbers of samples for an epoch. Epochs with less samples are removed.
    :return: np.ndarray or list of arrays
    """
    if type(trigger) == int:
        idx_triggers = np.where(lsl_trigger_decode(data_array[:, trigger_channel]) == trigger)[0]
        idx_start = np.concatenate([[0], idx_triggers])
        idx_end = np.concatenate([idx_triggers, [len(data_array)]])
    else:
        idx_start = np.where(lsl_trigger_decode(data_array[:, trigger_channel]) == trigger[0])[0]
        idx_end = np.where(lsl_trigger_decode(data_array[:, trigger_channel]) == trigger[1])[0]
        if len(idx_start) == 0:
            print(f"Trigger {trigger[0]} not found in given data.")
            idx_start = np.array([0])
        if len(idx_end) == 0:
            print(f"Trigger {trigger[1]} not found in given data.")
            idx_end = np.array([len(data_array)])

    if len(idx_start) > len(idx_end):
        idx_start = idx_start[:len(idx_end)]
    if len(idx_end) > len(idx_start):
        idx_end = idx_end[-len(idx_start):]

    keep = np.where((idx_end - idx_start) >= min_samples)[0]
    idx_start = idx_start[keep]
    idx_end = idx_end[keep]

    if cut_to_length == 'min':
        diff = idx_end - idx_start
        lmin = np.min(diff[diff != 0])
        idx_end = idx_start + lmin
    elif cut_to_length == 'max':
        lmax = np.max(idx_end - idx_start)
        idx_end = idx_start + lmax
    elif type(cut_to_length) == int:
        idx_end = idx_start + cut_to_length
        if idx_end[-1] > len(data_array):
            raise ValueError("Last epoch is not long enough to cut to {} samples.".format(cut_to_length))

    epochs = []
    for i_ep in range(len(idx_start)):
        ep = data_array[idx_start[i_ep]:idx_end[i_ep]]
        if remove_trigger_ch:
            ep = np.delete(ep, trigger_channel, axis=1)
        epochs.append(ep)

    if cut_to_length is None:
        return epochs
    else:
        return np.array(epochs)


class ADecoder(ABC):
    """Abstract decoder class.

    Defines the common interface for all decoders.
    """

    @staticmethod
    def _notch_filter(x, fs, exclude_channel=None):
        b, a = signal.iirnotch(w0=50., Q=30.0, fs=fs)
        if exclude_channel is None:
            return signal.filtfilt(b=b, a=a, x=x, axis=0)
        else:
            x_filt = signal.filtfilt(b=b, a=a, x=x, axis=0)
            x_filt[:, exclude_channel] = x[:, exclude_channel]
            return x_filt

    def __init__(self, lsl_inlet):
        """Constructor for fields which are common across all decoders.

        THIS IS AN ABSTRACT CLASS THUS IT CANNOT BE DIRECTLY INITIALISED

        """
        self.data = None
        self.epoch = None
        self.lsl_inlet = lsl_inlet
        self.clear_data()

    def pull_data(self, replace=True):
        new_data = pull_from_buffer(self.lsl_inlet)
        if replace or self.data is None:
            self.data = new_data
        else:
            self.data = np.vstack([self.data, new_data])
        return new_data

    def clear_data(self):
        pull_from_buffer(self.lsl_inlet)
        self.data = None


class AsynchronousDecoder(ADecoder):
    """Abstract decoder class that defines the common interface for all asynchronous decoders, i.e. decoders that only start decoding after the stmulus
    presentaiton has ended.
    """

    def __init__(self, lsl_inlet):
        super().__init__(lsl_inlet)

    @abc.abstractmethod
    def decode(self, paradigm_params, pull_new_data=True):
        pass

    def fit_calibration(self, trials_correct, stim_pres_order, paradigm_params, save_path=None):
        """
        Fitting method to be called during a calibration period (if this is necessary for the decoding method)
        :param trials_correct:
        :param stim_pres_order:
        :param paradigm_params:
        :param save_path:
        :return:
        """
        pass

    def fit_training(self, choice, save_path=None):
        """
        Fitting method to be called during a training period (if this is necessary for the decoding method
        :param choice: True choice among the presented stimuli
        :param save_path:
        :return:
        """
        pass

    def fit_resting(self, trig_rest_start, trig_rest_end, save_path=None):
        """
        Fitting method to be called during resting state period (if this is necessary for the decoding method
        :param save_path:
        :return:
        """
        pass

    def _get_last_epoch(self, trig_stim_start, trig_stim_end, notch_filter=True, remove_trigger_ch=False):
        self.epoch = cut_epochs(data_array=self.data,
                                trigger=(trig_stim_start, trig_stim_end),
                                remove_trigger_ch=remove_trigger_ch,
                                cut_to_length=None)[-1]
        if notch_filter:
            self.epoch = self._notch_filter(self.epoch, exclude_channel=None if remove_trigger_ch else -1)


class ContinuousDecoder(ADecoder):
    """Abstract decoder class that defines the common interface for all continuous decoders, i.e. decoders that
    continuously decode during stimulus presentation and that have to be running in a separate script.
    """

    def __init__(self, lsl_inlet, fs_eeg, n_eeg_channels, screen_freq, trigger_channel=-1):
        super().__init__(lsl_inlet)
        self._decoded = None
        self.n_eeg_channels = n_eeg_channels
        self.fs_eeg = fs_eeg
        self.screen_freq = screen_freq
        self.trigger_channel = trigger_channel
        self.model = None
        self._create_model()
        self.training_data = {'X': [],
                              'y': []}
        self._all_past_triggers = np.array([])

    def pull_data(self, replace=True):
        new_data = super().pull_data(replace=replace)
        if new_data is not None:
            self._all_past_triggers = np.concatenate([self._all_past_triggers,
                                                      lsl_trigger_decode(new_data[new_data[:, self.trigger_channel] != 0, self.trigger_channel])])
        return new_data

    def has_trig_happened(self, trig):
        self.pull_data(replace=False)
        if type(trig) == int:
            return trig in self._all_past_triggers
        elif type(trig) == list:
            for t in trig:
                if t in self._all_past_triggers:
                    return True
            return False
        else:
            raise AttributeError("Attribute trig must me int or list of ints.")

    @abc.abstractmethod
    def _create_model(self):
        pass

    @abc.abstractmethod
    def fit_model(self, epochs, batch_size, from_scratch=True, n_val=None, save_path=None):
        pass

    @abc.abstractmethod
    def decode(self, paradigm_params):
        pass

    @abc.abstractmethod
    def pull_training_data_X(self, paradigm_params):
        pass

    @abc.abstractmethod
    def set_training_data_y(self, y, paradigm_params):
        pass

    def wait_for_trig(self, triggers, wait_before_update_s: float = 0.2):
        if type(triggers) == int:
            triggers = [lsl_trigger_encode(triggers)]
        elif type(triggers) == list:
            triggers = [lsl_trigger_encode(t) for t in triggers]
        else:
            raise TypeError("Attribute triggers must be int or list of ints.")

        # pull initial data
        self.pull_data(replace=True)

        if self.data is None:
            raise ConnectionError("No EEG lsl stream. Connect EEG and check settings in config.py.")

        # pull with replacement until trig appears for the first time
        while np.sum([self.data[:, self.trigger_channel] == trig for trig in triggers]) == 0:
            time.sleep(wait_before_update_s)
            self.pull_data(replace=True)

    def _pull_data_until_nth_trig_plus_window(self, trig: int, n_trig:int, win_n_samples: int,
                                              wait_before_update_s: float = 0.02):
        """
        Continue to pull data from lsl inlet until there are at least win_n_samples present after the n_trig-th
        repetition of trigger trig.
        :param trig: trigger to be searched for
        :param n_trig: which repetition of the trigger to look out for
        :param win_n_samples: How many samples after the relevant trigger must be present
        :param wait_before_update_s: how long to wait before pulling from buffer again.
        :return: index of the n_trig-th occurence of trig in the data
        """
        trig = lsl_trigger_encode(trig)
        # pull until trig appears at least n_trig times:
        while np.sum(self.data[:, self.trigger_channel] == trig) < n_trig:
            time.sleep(wait_before_update_s)
            self.pull_data(replace=False)

        # index of the relevant trigger
        trig_idx = np.where(self.data[:, self.trigger_channel] == trig)[0][n_trig-1]

        # when there are not enough samples, pull again
        while self.data.shape[0] - trig_idx < win_n_samples:
            time.sleep(wait_before_update_s)
            self.pull_data(replace=False)

        return trig_idx


class CVEPDecoder(AsynchronousDecoder):
    """Decoder class for the CVEP paradigm class.

    """

    def __init__(self, lsl_inlet, fs_eeg, screen_freq, shift_crosscorr=1):
        """

        :param lsl_inlet:
        :param fs_eeg:
        :param screen_freq:
        :param shift_crosscorr:
        """

        super().__init__(lsl_inlet)
        self.lsl_inlet = lsl_inlet
        self.fs_eeg = fs_eeg
        self.screen_freq = screen_freq
        self.shift_crosscorr = shift_crosscorr
        self.template = None
        self.delay_order = None
        self.len_seq = None
        self.frame_shift = None
        self.len_template = None
        self.trig_stim_start = None
        self.trig_stim_end = None
        self.trig_stim_new_rep = None
        self.eeg_shift = None

    def fit_calibration(self, trials_correct, stim_pres_order, paradigm_params, save_path=None):
        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            return
        # extract paradigm parameters
        self.delay_order = paradigm_params['delay_order']
        self.len_seq = paradigm_params['len_seq'] * paradigm_params['frame_nr_multiplier']
        self.frame_shift = paradigm_params['frame_shift']
        self.len_template = int(self.len_seq / self.screen_freq * self.fs_eeg)
        self.trig_stim_start = paradigm_params['trig_stim_start']
        self.trig_stim_end = paradigm_params['trig_stim_end']
        self.trig_stim_new_rep = paradigm_params['trig_stim_new_rep']
        self.eeg_shift = self.frame_shift * self.fs_eeg / self.screen_freq

        # compute shift for each epoch
        shifts = (np.array(self.delay_order)[stim_pres_order] * self.eeg_shift).astype(int)

        with open(save_path[:-4] + "-data.pkl", 'wb') as f:
            pickle.dump(self.data, f)
        with open(save_path[:-4] + "-trials_correct.pkl", 'wb') as f:
            pickle.dump(trials_correct, f)

        # Cut epochs from data and choose only those which were correctly answered
        epochs = cut_epochs(data_array=self.data,
                            trigger=(self.trig_stim_start, self.trig_stim_end),
                            remove_trigger_ch=False,
                            cut_to_length=None)

        epochs_correct = [epochs[i] for i, c in enumerate(trials_correct) if c]

        # notch filter
        epoch_filt = [self._notch_filter(ep, exclude_channel=-1) for ep in epochs_correct]

        # Compute average shifted template for each stimulus
        templates = []
        for i, ep in enumerate(epoch_filt):
            reps = cut_epochs(data_array=ep,
                              trigger=self.trig_stim_new_rep,
                              remove_trigger_ch=True,
                              cut_to_length=self.len_template)

            reps_std = (reps - np.mean(reps, axis=1, keepdims=True)) / np.std(reps, axis=1, keepdims=True)
            mean_reps_shifted = np.roll(np.mean(reps_std, axis=0), - shifts[i], axis=0)
            templates.append(mean_reps_shifted)

        # Average over templates
        self.template = np.mean(np.array(templates), axis=0)

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.template, f)

    def decode(self, paradigm_params, pull_new_data=True):
        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            return

        # get data from buffer
        if pull_new_data:
            self.pull_data(replace=True)

        # cut out epoch and then cut into individual repetitions of pattern
        self._get_last_epoch(trig_stim_start=self.trig_stim_start,
                             trig_stim_end=self.trig_stim_end,
                             remove_trigger_ch=False,
                             notch_filter=True)
        repetitions = cut_epochs(data_array=self.epoch,
                                 trigger=self.trig_stim_new_rep,
                                 remove_trigger_ch=True,
                                 cut_to_length=self.len_template)

        # substract mean and divide by stdev, then compute mean
        reps_std = (repetitions - np.mean(repetitions, axis=1, keepdims=True)) / np.std(repetitions, axis=1, keepdims=True)
        mean_rep = np.mean(reps_std, axis=0)

        # compute cross-correlation with shifted template and averaged recorded data
        crosscorr = np.array([np.corrcoef([np.roll(self.template, sh, axis=0).flatten(), mean_rep.flatten()])[0, 1]
                              for sh in range(0, self.len_template, self.shift_crosscorr)])
        crosscor_max = np.max(crosscorr)
        idx_max = np.argmax(crosscorr)
        print("Maximum cross-correlation: {}".format(crosscor_max))

        # compute corresponding stimulus id
        delay_max = np.round(idx_max * self.shift_crosscorr * self.screen_freq / self.fs_eeg / self.frame_shift).astype(int)

        return np.where(self.delay_order == delay_max)[0][0]


class ShamDecoder(AsynchronousDecoder):
    """
    Decoder class for paradigms where the decoded stimulus id is already part of the dictionary returned by the .decode
    method, such as ClickSelection paradigm and continuous paradigms such as CVEPContinuous.

    """
    def __init__(self, lsl_inlet=None):
        super().__init__(lsl_inlet)

    def decode(self, paradigm_params, pull_new_data=True):
        return paradigm_params['decoded']


class LSLDecoder(AsynchronousDecoder):
    """
    Decoder class that receives the answer from a lsl stream. When .decode is called, it pulls one sample from the
    stream and returns this object as the decoded class.

    """
    def __init__(self, lsl_inlet, decoder_lsl_inlet):
        """
        :param lsl_inlet: pylsl.StreamInlet instance
        :param fs_eeg: sampling rate of the eeg stream
        """
        super().__init__(lsl_inlet)
        self.decoder_lsl_inlet = decoder_lsl_inlet

    def decode(self, paradigm_params, pull_new_data=True):
        return int(self.decoder_lsl_inlet.pull_chunk()[0][-1][0])


class SSVEPDecoder(AsynchronousDecoder):
    """Decoder class for the SSVEP paradigm class.
    """

    def __init__(self, lsl_inlet, fs_eeg, bandwidth=1.):
        super().__init__(lsl_inlet)
        self.lsl_inlet = lsl_inlet
        self.fs_eeg = fs_eeg
        self.bandwidth = bandwidth
        self.resting_data = None
        self.resting_spectrum = None
        self.mean_bandpower = None
        self.channel_weights = None
        self.frequencies = None
        self.freqs_unique = None

    def _spectrum(self, data, axis=0, n=None):
        if n is None:
            n = data.shape[axis]
        # log fft
        # spec = np.log(np.abs(np.fft.fft(data, axis=axis, n=n)))
        # freqs = np.fft.fftfreq(n=n, d=1 / self.fs_eeg)
        # welch
        freqs, spec = signal.welch(data, fs=self.fs_eeg, window='hann', nperseg=n, axis=axis)

        return spec, freqs

    def decode(self, paradigm_params, pull_new_data=True):
        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            return

        # get data from buffer
        if pull_new_data:
            self.pull_data(replace=True)

        self.frequencies = np.array(paradigm_params['frequencies'])  # shape (n_stim, n_freq_per_stim)
        self.freqs_unique = np.sort(np.unique(self.frequencies))
        n_freq_unique = len(self.freqs_unique)
        n_freq_per_stim = self.frequencies.shape[1]
        stim_duration_s = paradigm_params['stim_duration_s']
        eeg_frames_per_freq = int(stim_duration_s * self.fs_eeg)

        # cut out epoch and then cut into individual repetitions of pattern
        self._get_last_epoch(trig_stim_start=paradigm_params['trig_stim_start'],
                             trig_stim_end=paradigm_params['trig_stim_end'],
                             remove_trigger_ch=False,
                             notch_filter=True)

        # shape (n_freq_per_stim, eeg_frames_per_freq, n_eeg_channels)
        repetitions = cut_epochs(data_array=self.epoch,
                                 trigger=paradigm_params['trig_stim_new_rep'],
                                 remove_trigger_ch=True,
                                 cut_to_length=eeg_frames_per_freq)

        n_eeg_channels = repetitions.shape[-1]

        if self.channel_weights is None:
            self.channel_weights = np.ones((1, 1, n_eeg_channels), dtype=float)

        # compute spectrum for each repetition and each channel
        # shape of spectrum: (n_freq_per_stim, eeg_frames_per_freq, n_eeg_channels)
        spectrum, freqs = self._spectrum(repetitions, axis=1, n=None)

        # compute rs spectrum with same length as epochs in first call to decode
        if self.resting_spectrum is None:
            self.resting_spectrum, _ = self._spectrum(self.resting_data, axis=0, n=repetitions.shape[1])

        # substract resting state spectrum
        spectrum -= self.resting_spectrum

        # band pass filter and average in bands around frequencies
        frequencies_masks = np.array([(freqs >= f - self.bandwidth / 2) & (freqs <= f + self.bandwidth / 2)
                                      for f in self.freqs_unique])

        self.mean_bandpower = np.zeros((n_freq_unique, n_freq_per_stim, n_eeg_channels))
        for ifr in range(n_freq_unique):
            for irep in range(n_freq_per_stim):
                for ich in range(n_eeg_channels):
                    self.mean_bandpower[ifr, irep, ich] = np.mean(spectrum[irep, frequencies_masks[ifr], ich])

        max_idx = np.argmax(np.mean((self.mean_bandpower * self.channel_weights), axis=-1), axis=0)
        f_decoded = self.freqs_unique[max_idx]
        n_same = []
        for stim_freqs in self.frequencies:
            n_same.append(f_decoded == stim_freqs)
        n_same = np.array(n_same)

        # output index that best matches frequency pattern (with a bias to selecting earlier stimuli if none matches exactly)
        return np.argmax(np.sum(n_same, axis=-1))

    def fit_resting(self, trig_rest_start, trig_rest_end, save_path=None):
        """
        Fitting method to be called after resting state period (eyes open)
        :param save_path:
        :return:
        """
        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            return

        data = pull_from_buffer(self.lsl_inlet)

        epoch = cut_epochs(data_array=data,
                           trigger=(trig_rest_start, trig_rest_end),
                           remove_trigger_ch=True,
                           cut_to_length=None)[-1]
        epoch = self._notch_filter(epoch, exclude_channel=None)

        self.resting_data = epoch

    def fit_training(self, choice, save_path=None):
        """
        Fitting method to be called during training periods.
        :param choice: True choice among the presented stimuli
        :param save_path:
        :return:
        """
        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            return

        n_eeg_ch = self.channel_weights.shape[-1]
        max_idxs = np.argmax(self.mean_bandpower, axis=0)  # shape (n_freq_per_stim, n_eeg_channels)
        f_decoded = self.freqs_unique[max_idxs]

        n_same_per_ch = []
        for ch in range(max_idxs.shape[-1]):
            n_same_per_ch.append(np.sum(f_decoded[:, ch] == self.frequencies[choice]))
        n_same_per_ch = np.array(n_same_per_ch, dtype=float)

        learning_exponent = 1.2
        self.channel_weights = self.channel_weights * learning_exponent**(2 * n_same_per_ch - n_eeg_ch)
        self.channel_weights = self.channel_weights / np.mean(self.channel_weights)
        print(f"Updated channel weights: {self.channel_weights[0,0]}")


class CVEPContinuousDecoder(ContinuousDecoder):
    """Decoder class for the CVEPContinuous paradigm class. An instance of this decoder must be passed when
    instanciating the CVEPContinuous class.
    """

    def __init__(self, lsl_inlet, fs_eeg, n_eeg_channels, screen_freq, trigger_channel, eeg_window_len_ms,
                 n_stim_min, classification_thresh):
        """

        :param lsl_inlet:
        :param fs_eeg:
        :param screen_freq:
        """
        self.eeg_window_len_ms = eeg_window_len_ms
        self.eeg_window_len_samples = int(self.eeg_window_len_ms / 1000 * fs_eeg)
        super().__init__(lsl_inlet, fs_eeg, n_eeg_channels, screen_freq, trigger_channel)
        self.n_stim_min = n_stim_min
        self.classification_thresh_min = classification_thresh[0]
        self.classification_thresh_max = classification_thresh[1]
        if self.trigger_channel < 0:
            tc = self.n_eeg_channels + 1 + self.trigger_channel
        else:
            tc = self.trigger_channel
        self.data_channels = np.arange(self.n_eeg_channels+1) != tc

    def _create_model(self):
        self.model = cvep_cnn_nagel2019(eeg_sampling_rate=self.fs_eeg,
                                        window_size_samples=self.eeg_window_len_samples,
                                        nr_eeg_channels=self.n_eeg_channels,
                                        pretrained=False)

    def save_test_data(self, filename, trig_idx_all):
        with open(filename, "wb") as f:
            pickle.dump(self.data[trig_idx_all[0]:trig_idx_all[-1]+self.eeg_window_len_samples,
                        self.data_channels].reshape(1, -1, self.n_eeg_channels), f)
        with open(filename[:-4] + "_trigs.pkl", "wb") as f:
            pickle.dump(np.array(trig_idx_all)-trig_idx_all[0], f)

    def fit_model(self, epochs, batch_size, from_scratch=True, n_val=None, save_path=None, verbosity=1):
        if from_scratch:
            self._create_model()

        X_train = np.array(self.training_data['X'])
        y_train = np.stack([-np.array(self.training_data['y']) + 1, np.array(self.training_data['y'])], axis=1)
        data_val = None

        if n_val is not None:
            X_train = X_train[:n_val]
            y_train = y_train[:n_val]
            X_val = X_train[n_val:]
            y_val = y_train[n_val:]
            data_val = (X_val, y_val)

        h = self.model.fit(x=X_train, y=y_train, validation_data=data_val, epochs=epochs, batch_size=batch_size,
                           verbose=verbosity)

        if save_path is not None:
            self.model.save(filepath=save_path)

        return h

    def load_fitted_model(self, path: str):
        self.model = keras.models.load_model(path)

    def decode(self, paradigm_params, wait_before_update_s=0.01, special_stim_id=None, special_stim_frame_wait=10,
               save_path=None):
        """
        To be called by separate python script than the experiment main.
        :param paradigm_params: Dictionary returned by present method of corresponding paradigm
        :param wait_before_update_s: How long to wait between initial data pulling
        :param special_stim_id: Index of a special stimulus that should have a harder threshold before being accepted
         (such as an exit button)
         :param special_stim_frame_wait: How many times the special stimulus has to clear the acceptance threshold
         before being accepted.
        :return:
        """
        stim_sequences = paradigm_params['stim_sequences']
        trig_stim_start = paradigm_params['trig_stim_start']
        trig_stim_new_rep = paradigm_params['trig_stim_new_rep']
        frame_nr_multiplier = paradigm_params['frame_nr_multiplier']
        max_repetitions = paradigm_params['max_repetitions']

        sequence_len = int(len(stim_sequences) / frame_nr_multiplier)
        stim_sequences_frames_long = np.vstack([stim_sequences[::frame_nr_multiplier]]*max_repetitions)

        # reset the decoded variable
        self._decoded = None

        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            print("No eeg connected, not decoding.")
            return

        self.wait_for_trig(triggers=trig_stim_start,
                           wait_before_update_s=0.2)

        decoded_frames = []
        stim_nr = 0
        trig_idx_all = []

        special_stim_count = 0
        while stim_nr < len(stim_sequences_frames_long):
            # pull data until the n-th trigger and window is there
            trig_idx = self._pull_data_until_nth_trig_plus_window(trig=trig_stim_new_rep,
                                                                  n_trig=stim_nr+1,
                                                                  win_n_samples=self.eeg_window_len_samples,
                                                                  wait_before_update_s=wait_before_update_s)

            trig_idx_all.append(trig_idx)

            # make prediction
            pred = self.model.predict(self.data[trig_idx:trig_idx+self.eeg_window_len_samples,
                                                     self.data_channels].reshape(1, self.eeg_window_len_samples, self.n_eeg_channels),
                                      verbose=0)
            decoded_frames.append(np.argmax(pred[0]))

            # compare prediction with actual sequences (but wait for minimum sequence length)
            if stim_nr >= self.n_stim_min:
                # check which decoded frames match the stimulation sequences where
                match_where = np.array(decoded_frames).reshape(-1, 1) == stim_sequences_frames_long[:len(decoded_frames)]

                # take into account only the last sequence_len frames and compute accuracy per stimulus
                acc = np.mean(match_where[-sequence_len:], axis=0)

                # make prediction if the acc of one stimulus is higher than thresh max and all others are
                # below thresh min
                if np.sum(acc >= self.classification_thresh_max) == 1:
                    if np.sum(acc < self.classification_thresh_min) == stim_sequences_frames_long.shape[1]-1:
                        self._decoded = np.argmax(acc)

                        if self._decoded != special_stim_id:
                            if save_path is not None:
                                self.save_test_data(filename=save_path, trig_idx_all=trig_idx_all)
                            return self._decoded
                        else:
                            special_stim_count += 1
                            if special_stim_count >= special_stim_frame_wait:
                                if save_path is not None:
                                    self.save_test_data(filename=save_path, trig_idx_all=trig_idx_all)
                                return self._decoded
                            self._decoded = None

            stim_nr += 1
        print(acc)
        self._decoded = np.argmax(acc)
        if save_path is not None:
            self.save_test_data(filename=save_path, trig_idx_all=trig_idx_all)
        return self._decoded

    def pull_training_data_X(self, paradigm_params, wait_before_update_s=0.5, save_path=None):

        # Makes it possible to run experiment without eeg data for testing by setting lsl_inlet to None
        if self.lsl_inlet is None:
            print("No eeg connected, not calibrating.")
            return

        trig_stim_start = paradigm_params['trig_stim_start']
        trig_stim_new_rep = paradigm_params['trig_stim_new_rep']
        trig_stim_end = paradigm_params['trig_stim_end']

        # # wait for trig_stim_start
        # self.wait_for_trig(triggers=trig_stim_start,
        #                    wait_before_update_s=wait_before_update_s)

        # pull data until trig_stim_end + window
        self._pull_data_until_nth_trig_plus_window(trig=trig_stim_end,
                                                   n_trig=1,
                                                   win_n_samples=self.eeg_window_len_samples,
                                                   wait_before_update_s=wait_before_update_s)

        # format data and add it to self.training_data['X']
        idxs = np.where(self.data[:, self.trigger_channel] == lsl_trigger_encode(trig_stim_new_rep))[0]
        if self.trigger_channel < 0:
            tc = self.data.shape[1] + self.trigger_channel
        else:
            tc = self.trigger_channel
        data_channels = np.arange(self.data.shape[1]) != tc
        for i in idxs:
            self.training_data['X'].append(self.data[i:i+self.eeg_window_len_samples, data_channels])

        self.clear_data()

        if save_path is not None:
            print("...saving training data (X)")
            with open(save_path, "wb") as f:
                pickle.dump(self.training_data['X'], f)

    def set_training_data_y(self, y, paradigm_params, save_path=None):
        stim_sequences = paradigm_params['stim_sequences']
        frame_nr_multiplier = paradigm_params['frame_nr_multiplier']
        reps = paradigm_params['max_repetitions']

        self.training_data['y'] += stim_sequences[::frame_nr_multiplier, y].tolist() * reps

        assert len(self.training_data['X']) == len(self.training_data['y'])

        if save_path is not None:
            print("...saving training data (y)")
            with open(save_path, "wb") as f:
                pickle.dump(self.training_data['y'], f)



