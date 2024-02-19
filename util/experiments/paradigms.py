import abc
import os.path
from abc import ABC
from typing import Dict, Any, Collection

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
import psychopy.visual
# from psychopy import prefs
# prefs.hardware['audioLib'] = ['ptb']
from psychopy import core, event, parallel
from shared_memory_dict import SharedMemoryDict

from util.experiments.standard import send_trig_if_parport
from util.helper import possible_freqs


class AParadigm(ABC):
    """Abstract paradigm class.

    Defines the common interface for all Paradigms.
    """

    def __init__(self, window: psychopy.visual.Window, objects_stim, objects_steady,
                 trig_stim_start: int = None, trig_stim_end: int = None, parport: parallel.ParallelPort = None,
                 show_obj_stim_when_waiting=True):
        """Constructor for fields which are ccommon across all paradigms.

        THIS IS AN ABSTRACT CLASS THUS IT CANNOT BE DIRECTLY INITIALISED

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which will be changed during the
        paradigm presentaiton (e.g. flashed)
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; a collection of psychopy stimuli which
        don't change and are drawn every frame
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        """
        self.window = window
        self.objects_stim = objects_stim
        if type(self.objects_stim) == list:
            self.n_stim = len(self.objects_stim)
        elif type(self.objects_stim) == psychopy.visual.elementarray.ElementArrayStim:
            self.n_stim = self.objects_stim.nElements
        else:
            raise AttributeError("objects_stim must be list or psychopy.visual.elementarray.ElementArrayStim")

        if objects_steady is None:
            self.objects_steady = []
        elif type(objects_steady) is not list:
            self.objects_steady = [objects_steady]
        else:
            self.objects_steady = objects_steady

        self.trig_stim_start = trig_stim_start
        self.trig_stim_end = trig_stim_end
        self.parport = parport
        self.show_obj_stim_when_waiting = show_obj_stim_when_waiting

    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Returns a dictionary containing all parameters of the paradigm. The results of this method should
        be returned by `present()`. Any information that a decoder needs must be returned by this method.

        :return: a dictionary mapping a string naming a given parameter to its value
        """
        pass

    @abc.abstractmethod
    def present(self, repetitions=None, save_frame_intervals_path=None) -> None:
        """Start the stimulus presentation

        :return: only the parameter dict (via self.get_params)
        """
        pass

    @staticmethod
    def _change_color(stimuli, new_color, condition=lambda object_nr: True, colorSpace='rgb') -> None:
        """Changes the color of the given stimuli. Also accept a condition parameter, which filters which stimuli
        will be affected

        :param stimuli: a collection of stimuli to be changed
        :param new_color: the color the stimuli will be set to. See https://www.psychopy.org/general/colours.html.
        :param condition: a function which returns True/False given an object number, determining whether the object
        will change color
        :return: nothing
        """
        if type(stimuli) == list:
            for i, obj in enumerate(stimuli):
                if condition(i):
                    obj.setColor(new_color, colorSpace=colorSpace)
        elif type(stimuli) == psychopy.visual.elementarray.ElementArrayStim:
            colors = [new_color if condition(i) else stimuli.color for i in range(stimuli.nElements)]
            stimuli.setColors(colors=colors)

    def _draw_steady(self) -> None:
        """Draws all steady stimuli on the screen

        :return: nothing
        """
        for obj in self.objects_steady:
            obj.draw()

    def _draw_stim(self, frame_states: np.ndarray = None) -> None:
        if type(self.objects_stim) == list:
            if frame_states is None:  # draw all
                for obj in self.objects_stim:
                    obj.draw()
            else:  # draw according to frame state rule
                for i, obj in enumerate(self.objects_stim):
                    if frame_states[i]:
                        obj.draw()
        elif type(self.objects_stim) == psychopy.visual.elementarray.ElementArrayStim:
            if frame_states is None:
                frame_states = 1.
            self.objects_stim.setOpacities(frame_states)
            self.objects_stim.draw()
        else:
            raise NotImplementedError

    def _draw_frame(self, flip_window: bool = True) -> None:
        """Draws all steady stimuli on screen, and stim stimuli only if self.show_obj_stim_when_waiting is True.
        May call window.flip() to draw the frame
        CAREFUL, FLIPPING THE WINDOW MAKES THE PROGRAM WAIT UNTIL THE FRAME IS DRAWN.

        :param flip_window: bool, determines whether the window will be flipped at the end of the call or not
        :return: nothing
        """
        self._draw_steady()
        if self.show_obj_stim_when_waiting:
            self._draw_stim()
        if flip_window:
            self.window.flip()

    def _send_trigger(self, start=True):
        if start:
            trig = self.trig_stim_start
        else:
            trig = self.trig_stim_end
        send_trig_if_parport(trigger=trig, parport=self.parport)


class Oddball(AParadigm):
    """TODO: A description of the oddball paradigm

    """

    def __init__(self, window: psychopy.visual.Window, objects_stim: Collection, objects_steady: Collection,
                 trig_stim_start: int = None, trig_stim_end: int = None, parport: parallel.ParallelPort = None,
                 show_obj_stim_when_waiting=True,
                 rep_per_object: int = 5, simultaneous: int = 2, blink_duration: float = 0.05, blink_distance: float = 0.2,
                 random_seed: int = None, color_blink_on=(1., 1., 1.), color_blink_off=(0.5, 0.5, 0.5), wait_before_blink: float = None
                 ):
        """Constructor for the Oddball paradigm.

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which will be flashed during the
        paradigm presentaiton
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; (a collection of) psychopy stimuli which
        don't change and are drawn every frame
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        :param rep_per_object: int; how often each simulus is flashed
        :param simultaneous: int; how many stimuli are flashed at the same time
        :param blink_duration: float; in seconds, defines how long a blink stays on screen
        :param blink_distance: float; in seconds, how much time passes between the stimulus flashes
        :param random_seed: int; random seed used when shuffling the stimulation order
        :param color_blink_on: the on color of objects_stim. See https://www.psychopy.org/general/colours.html.
        :param color_blink_off: the off color of objects_stim. See https://www.psychopy.org/general/colours.html.
        :param wait_before_blink: float; the wait time at the beginning of the experiment
        """
        super().__init__(window=window, objects_stim=objects_stim, objects_steady=objects_steady,
                         trig_stim_start=trig_stim_start, trig_stim_end=trig_stim_end, parport=parport,
                         show_obj_stim_when_waiting=show_obj_stim_when_waiting)
        self.rep_per_object = rep_per_object
        self.simultaneous = simultaneous
        self.blink_duration = blink_duration
        self.blink_distance = blink_distance
        self._random_seed = random_seed
        self.color_blink_on = color_blink_on
        self.color_blink_off = color_blink_off
        self.wait_before_blink = wait_before_blink
        self.stim_order = None

    def get_params(self):
        """Returns a dictionary containing all parameters of the paradigm. The results of this method should
        be returned by `present()`. Any information that a decoder needs must be returned by this method.

        :return: a dictionary mapping a string naming a given parameter to its value
        """
        params = {'rep_per_object': self.rep_per_object,
                  'simultaneous': self.simultaneous,
                  'blink_duration': self.blink_duration,
                  'blink_distance': self.blink_distance,
                  'color_blink_on': self.color_blink_on,
                  'color_blink_off': self.color_blink_off,
                  'wait_before_blink': self.wait_before_blink,
                  'trig_stim_start': self.trig_stim_start,
                  'trig_stim_end': self.trig_stim_end,
                  'n_stim': self.n_stim,
                  'stim_order': self.stim_order
                  }
        return params

    def _generate_stim_order(self):
        np.random.seed(self._random_seed)
        ids = np.arange(self.n_stim)

        blink = []
        for rep in range(self.rep_per_object):
            np.random.shuffle(ids)
            blink.append(ids.reshape(-1, self.simultaneous).copy())

        self.stim_order = np.vstack(blink)
        self._random_seed += 1

    def present(self, repetitions=None, save_frame_intervals_path=None):
        self._generate_stim_order()

        # time to read the options
        if self.wait_before_blink is not None:
            self._change_color(self.objects_stim, self.color_blink_on)
            self._draw_frame()
            core.wait(self.wait_before_blink)

        # initial frame all grey
        self._change_color(self.objects_stim, self.color_blink_off)
        self._draw_frame()

        self._send_trigger(start=True)

        for stim_on in self.stim_order:
            self._change_color([self.objects_stim[i] for i in range(self.n_stim) if i in stim_on], self.color_blink_on)
            self._draw_frame()
            core.wait(self.blink_duration)

            # all grey
            self._change_color([self.objects_stim[i] for i in range(self.n_stim) if i in stim_on], self.color_blink_off)
            self._draw_frame()
            core.wait(self.blink_distance - self.blink_duration)

        self._send_trigger(start=False)

        # final frame show all in white
        self._change_color(self.objects_stim, self.color_blink_on)
        self._draw_frame()

        return self.get_params()


class CVEP(AParadigm):
    """ TODO: Describe CVEP

    """

    @staticmethod
    def _generate_code_sequence(len_seq: int) -> np.ndarray:
        """
        :param len_seq: defines how long the generated code sequence is
        :return: returns a numpy array of the sequence of 0 and 1.
        """
        base = np.log2(len_seq + 1)
        if base != np.round(base):
            raise ValueError('len_seq must be an integer that equals 2^n - 1 for an integer n.')
        return scipy.signal.max_len_seq(int(base))[0]

    @staticmethod
    def _create_circular_shifts(code_sequence: np.ndarray, frame_shift: int, n_shifts: int) -> np.ndarray:
        """
        :param code_sequence: np.ndarray; sequence of 0 and 1.
        :param frame_shift: int; defines the delta between two shifts
        :param n_shifts: int; how many shifts are created (n_shifts is typically the number of stimuli)
        :return: 2D np.ndarray of dimension (len(code_sequence), n_shifts)
        """
        return np.array([np.roll(code_sequence, i * frame_shift) for i in range(n_shifts)]).T

    def __init__(self, window: psychopy.visual.Window, objects_stim: Collection, objects_steady: Collection,
                 trig_stim_start: int = None, trig_stim_end: int = None, trig_stim_new_rep: int = None,
                 parport: parallel.ParallelPort = None, show_obj_stim_when_waiting=True,
                 wait_before_blink: int = None, delay_order: Collection = None,
                 len_seq: int = 63, frame_shift: int = 6, frame_nr_multiplier: int = 1, repetitions: int = 1,
                 color_blink=(1.,1.,1.)):
        """Constructs the CVEP paradigm

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which will be flashed according to
        their respective code sequence
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; a collection of psychopy stimuli which
        don't change and are drawn every frame
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param trig_stim_new_rep: int; Trigger to be sent after each repetition
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        :param wait_before_blink: float; in seconds, the wait time at the beginning of the experiment
        :param delay_order: [int] or None; order in which the stimuli are delayed. E.g. [0,2,1,...] means
        that the first stimulus is presented with the unshifted sequence, the second is presented with the twice-shifted
        and the third stimulus is presented with the once shifted sequence. If None, the delay order is the same as the
        stimulus order.
        :param len_seq: int; length of the code sequence. Must be 2^n - 1.
        :param frame_shift: int; defines the delta between two shifts of the code sequence. Should ideally be a
        multiple of the denominator of the ratio of eeg sampling frequency and monitor frequency.
        :param frame_nr_multiplier: int; number of frames that each element of the sequence is shown on the screen.
        :param repetitions: int; number of times the sequence will be repeated in each stimulus presentation.
        :param color_blink: Color that the stimuli are shown in. If None, the color is not changed
        """

        super().__init__(window=window, objects_stim=objects_stim, objects_steady=objects_steady,
                         trig_stim_start=trig_stim_start, trig_stim_end=trig_stim_end, parport=parport,
                         show_obj_stim_when_waiting=show_obj_stim_when_waiting)
        self.len_seq = len_seq
        self.frame_shift = frame_shift
        self.delay_order = delay_order
        self.wait_before_blink = wait_before_blink
        self.color_blink = color_blink
        self.frame_nr_multiplier = frame_nr_multiplier
        self.repetitions = repetitions
        self.trig_stim_new_rep = trig_stim_new_rep
        self._fame_intervals = []

        if self.n_stim * self.frame_shift > self.len_seq * self.frame_nr_multiplier:
            raise ValueError('len(objects_stim) * frame_shift must be less or equal to len_seq * frame_nr_multiplier.')

        self.msequence = self._generate_code_sequence(self.len_seq)
        self.stim_sequences = self._create_circular_shifts(np.repeat(self.msequence, self.frame_nr_multiplier),
                                                           self.frame_shift, self.n_stim)
        if self.delay_order is not None:
            self.stim_sequences = self.stim_sequences[:, self.delay_order]

    def get_params(self):
        """Returns a dictionary containing all parameters of the paradigm. The results of this method should
        be returned by `present()`. Any information that a decoder needs must be returned by this method.

        :return: a dictionary mapping a string naming a given parameter to its value
        """
        params = {'delay_order': self.delay_order,
                  'n_stim': self.n_stim,
                  'len_seq': self.len_seq,
                  'frame_shift': self.frame_shift,
                  'frame_nr_multiplier': self.frame_nr_multiplier,
                  'msequence': self.msequence,
                  'stim_sequences': self.stim_sequences,
                  'wait_before_blink': self.wait_before_blink,
                  'trig_stim_start': self.trig_stim_start,
                  'trig_stim_end': self.trig_stim_end,
                  'trig_stim_new_rep': self.trig_stim_new_rep
                  }
        return params

    def _presentation_preparation(self, save_frame_intervals_path):
        # set all stim objects to the same color (in case this was not the case before presentation)
        if self.color_blink is not None:
            self._change_color(self.objects_stim, self.color_blink)

        # render all steady stimuli into one image stimulus
        screenshot = psychopy.visual.BufferImageStim(self.window, stim=self.objects_steady)

        # time to read the options
        if self.wait_before_blink is not None:
            self._draw_frame()
            core.wait(self.wait_before_blink)

        if save_frame_intervals_path is not None:
            self.window.recordFrameIntervals = True
            self._frame_intervals = []

        # stimulation
        self._send_trigger(start=True)
        self._draw_frame()

        return screenshot

    def _presentation_end(self, save_frame_intervals_path):
        # final frame show all
        self._draw_frame()
        self._send_trigger(start=False)

        if save_frame_intervals_path is not None:
            self.window.recordFrameIntervals = False
            self.window.frameIntervals = []
            self.window.frameClock.reset()
            plt.plot(np.array(self._frame_intervals).T)
            plt.savefig(save_frame_intervals_path + ".png")

            with open(save_frame_intervals_path + ".pkl", 'wb') as f:
                pickle.dump(np.array(self._frame_intervals), f)

    def present(self, repetitions=None, save_frame_intervals_path=None):
        if repetitions is None:
            repetitions = self.repetitions

        screenshot = self._presentation_preparation(save_frame_intervals_path=save_frame_intervals_path)

        for rep in range(repetitions):
            send_trig_if_parport(trigger=self.trig_stim_new_rep, parport=self.parport)

            for frame_nr, frame_states in enumerate(self.stim_sequences):
                screenshot.draw()
                self._draw_stim(frame_states=frame_states)
                self.window.flip()

            if save_frame_intervals_path is not None:
                # frame intervals per repetition
                self._frame_intervals.append(self.window.frameIntervals)
                self.window.frameIntervals = []
                self.window.frameClock.reset()

        self._presentation_end(save_frame_intervals_path=save_frame_intervals_path)

        return self.get_params()


class CVEPContinuous(CVEP):
    def __init__(self, window: psychopy.visual.Window, objects_stim: Collection, objects_steady: Collection,
                 trig_stim_start: int = None, trig_stim_end: int = None, trig_stim_new_rep: int = None,
                 parport: parallel.ParallelPort = None, show_obj_stim_when_waiting=True,
                 wait_before_blink: int = None, delay_order: Collection = None,
                 len_seq: int = 63, frame_shift: int = 6, frame_nr_multiplier: int = 1, max_repetitions: int = 1,
                 color_blink=(1., 1., 1.), shared_dict_name="shared_dict_0"):
        """Constructs the CVEP paradigm for simultaneous presentation and decoding. For A CVEP paradigm which just
        presents the flicker sequence (for decoding afterwards), use the class CVEP.

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which will be flashed according to
        their respective code sequence
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; a collection of psychopy stimuli which
        don't change and are drawn every frame
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param trig_stim_new_rep: int; Trigger to be sent after each repetition
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        :param wait_before_blink: float; in seconds, the wait time at the beginning of the experiment
        :param delay_order: [int] or None; order in which the stimuli are delayed. E.g. [0,2,1,...] means
        that the first stimulus is presented with the unshifted sequence, the second is presented with the twice-shifted
        and the third stimulus is presented with the once shifted sequence. If None, the delay order is the same as the
        stimulus order.
        :param len_seq: int; length of the code sequence. Must be 2^n - 1.
        :param frame_shift: int; defines the delta between two shifts of the code sequence. Should ideally be a
        multiple of the denominator of the ratio of eeg sampling frequency and monitor frequency.
        :param frame_nr_multiplier: int; number of frames that each element of the sequence is shown on the screen.
        :param max_repetitions: int; maximum number of times the sequence will be repeated in each stimulus
        presentation, before giving up. If None, the sequence will be presented until a result is decoded or the
        escape key is pressed.
        :param color_blink: Color that the stimuli are shown in. If None, the color is not changed
        """

        super().__init__(window=window, objects_stim=objects_stim, objects_steady=objects_steady,
                         trig_stim_start=trig_stim_start, trig_stim_end=trig_stim_end,
                         trig_stim_new_rep=trig_stim_new_rep, parport=parport,
                         show_obj_stim_when_waiting=show_obj_stim_when_waiting,
                         wait_before_blink=wait_before_blink, delay_order=delay_order, len_seq=len_seq,
                         frame_shift=frame_shift, frame_nr_multiplier=frame_nr_multiplier, repetitions=1,
                         color_blink=color_blink
                         )
        self.max_repetitions = max_repetitions
        self._decoded = None
        self.shared_dict = SharedMemoryDict(name=shared_dict_name, size=1024*2)
        self.shared_dict['decode_params'] = {'stim_sequences': self.stim_sequences,
                                             'max_repetitions': self.max_repetitions,
                                             'trig_stim_start': self.trig_stim_start,
                                             'trig_stim_end': self.trig_stim_end,
                                             'trig_stim_new_rep': self.trig_stim_new_rep,
                                             'frame_nr_multiplier': self.frame_nr_multiplier}

        self.shared_dict['decoded'] = None

    def get_params(self):
        """Returns a dictionary containing all parameters of the paradigm. The results of this method should
        be returned by `present()`. Any information that a decoder needs must be returned by this method.

        :return: a dictionary mapping a string naming a given parameter to its value
        """
        params = super().get_params()
        params['decoded'] = self._decoded
        params['max_repetitions'] = self.max_repetitions
        return params

    def present(self, repetitions=None, save_frame_intervals_path=None):
        screenshot = self._presentation_preparation(save_frame_intervals_path=save_frame_intervals_path)
        self.shared_dict['decoded'] = None
        self._decoded = None

        frame_nr = 0
        full_seq_len = len(self.stim_sequences)
        while frame_nr / full_seq_len < self.max_repetitions and self._decoded is None:
            if frame_nr % self.frame_nr_multiplier == 0:
                send_trig_if_parport(trigger=self.trig_stim_new_rep, parport=self.parport)

            # present frame
            screenshot.draw()
            self._draw_stim(frame_states=self.stim_sequences[frame_nr % full_seq_len])
            self.window.flip()

            # get result or None from decoder
            self._decoded = self.shared_dict['decoded']
            frame_nr += 1

        if save_frame_intervals_path is not None:
            self._frame_intervals.append(self.window.frameIntervals)

        self._presentation_end(save_frame_intervals_path=save_frame_intervals_path)
        if self._decoded is None:
            core.wait(0.3)
            self._decoded = self.shared_dict['decoded']
        return self.get_params()


class ClickSelection(AParadigm):
    """ TODO: Explain ClickSelection paradigm

    """

    def __init__(self, window: psychopy.visual.Window, objects_stim: Collection, objects_steady: Collection,
                 trig_stim_start: int = None, trig_stim_end: int = None, parport: parallel.ParallelPort = None,
                 show_obj_stim_when_waiting=True):
        """Constructs a ClickSelection paradigm.

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which can be selcted by clicking
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; a collection of psychopy stimuli which
        don't change and are not clickable
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        """
        super().__init__(window=window, objects_stim=objects_stim, objects_steady=objects_steady,
                         trig_stim_start=trig_stim_start, trig_stim_end=trig_stim_end, parport=parport,
                         show_obj_stim_when_waiting=show_obj_stim_when_waiting)
        if type(self.objects_stim) != list:
            raise AttributeError("objects_stim must be a list of stimulus objects for the click paradigm.")

        self.clicked = None
        self.mouse = event.Mouse(visible=False,
                                 newPos=(0, 0),
                                 win=self.window)

    def get_params(self):
        params = {'decoded': self.clicked,
                  'trig_stim_start': self.trig_stim_start,
                  'trig_stim_end': self.trig_stim_end
                  }
        return params

    def present(self, repetitions=None, save_frame_intervals_path=None):
        self._change_color(self.objects_stim, (1., 1., 1.))
        self._draw_frame()
        self._send_trigger(start=True)

        self.mouse.setVisible(True)
        self.mouse.setPos((0, 0))

        print("Waiting for click selection...")
        self.clicked = None
        while self.clicked is None:
            for i, obj in enumerate(self.objects_stim):
                if self.mouse.isPressedIn(obj, buttons=[0]):
                    self.clicked = i

        self._send_trigger(start=False)
        self.mouse.setVisible(False)

        return self.get_params()


class SSVEP(AParadigm):
    """TODO: Describe SSVEP

    """

    @staticmethod
    def _to_np_array(frequencies) -> np.ndarray:
        if type(frequencies) is list:
            return np.array(frequencies).reshape(len(frequencies), -1)
        if type(frequencies) is np.ndarray:
            return frequencies.reshape(len(frequencies), -1)
        raise ValueError('frequencies must be list or numpy array.')

    @staticmethod
    def _get_frames_from_frequencies(frequencies, monitor_freq, stim_duration_s) -> np.ndarray:
        """Computes the sequences of on / off frames for the given frequencies, monitor frequency and stimulation
        duration.

        :param frequencies: np.array of frequencies which will be displayed. Must be of shape (n_stimuli, n_freqs_per_stim)
        :param monitor_freq: the refresh rate of the screen
        :param stim_duration_s: the stimulation duration (per frequency) in seconds
        :return: np.ndarray of frames
        """
        poss_freqs, frame_dist = possible_freqs(monitor_freq)
        unique_freqs = np.unique(frequencies.flatten())
        dict_framedist = {}
        for f in unique_freqs:
            if f in poss_freqs:
                dict_framedist[f] = frame_dist[poss_freqs == f][0]
            else:
                raise ValueError(f"A frequency of {f} Hz cannot be displayed correctly by this monitor of refresh rate "
                                 f"{monitor_freq} Hz.")

        frames = np.zeros((*frequencies.shape, int(monitor_freq * stim_duration_s) ))  # shape: (n_stim, n_freq_per_stim, n_frames)
        for f in unique_freqs:
            this_frames = np.zeros(int(monitor_freq * stim_duration_s))
            dist = dict_framedist[f]
            fnr = int(dist / 2)

            for i in range(fnr):
                this_frames[i::dist] = 1

            frames[frequencies == f] = this_frames

        return frames.astype(int).reshape(frequencies.shape[0], -1)

    def __init__(self, window: psychopy.visual.Window, objects_stim: Collection, objects_steady: Collection,
                 frequencies: Collection, trig_stim_start: int = None, trig_stim_end: int = None,
                 trig_stim_new_rep: int = None, parport: parallel.ParallelPort = None,
                 show_obj_stim_when_waiting=True, monitor_freq: int = 60,
                 stim_duration_s: float = 1., shuffle_freqs: bool = False, random_seed: int = None,
                 wait_before_blink: int = None, color_blink=(1., 1., 1.)
                 ):
        """Constructs the SSVEP paradigm

        :param window: psychopy.visual.window; window on which the paradigm will be run
        :param objects_stim: [psychopy stimulus] ; collection of psychopy stimuli which will be flashed during the
        paradigm presentaiton
        :param objects_steady: [psychopy stimulus] or single psychopy stimulus; (a collection of) psychopy stimuli which
        don't change and are drawn every frame
        :param frequencies: np.ndarray or list of floats; a collection of frequencies to be displayed via object_stim
        stimuli. Must be same length as object_stim. May optionally have a second dimension if each object should be
        displayed at more than one frequency in succession (then the shape is (n_stim, n_freq_per_stimulus).
        :param trig_stim_start: int; Trigger to be sent when starting stimulus presentation
        :param trig_stim_end: int; Trigger to be sent directly after stimulus presentation
        :param parport: psychopy.parallel.ParallelPort; parallel port object to send the triggers with. If None,
        no triggers are sent.
        :param monitor_freq: int; the refresh rate of the used monitor
        :param stim_duration_s: float; the duration of the stimulus presentation in seconds
        :param shuffle_freqs: bool; indicating whether the frequencies are shuffled
        :param random_seed: an integer used as the seed when shuffling the frequencies
        :param wait_before_blink: the wait time at the beginning of the presentation
        :param color_blink: Color the stimuli will be flashed in. If None, the color is not changed.
        """

        super().__init__(window=window, objects_stim=objects_stim, objects_steady=objects_steady,
                         trig_stim_start=trig_stim_start, trig_stim_end=trig_stim_end, parport=parport,
                         show_obj_stim_when_waiting=show_obj_stim_when_waiting)
        self.trig_stim_new_rep = trig_stim_new_rep

        if len(frequencies) != self.n_stim:
            raise AttributeError("frequencies must be of same length as object_stim")

        self.frequencies = self._to_np_array(frequencies)
        self.n_freqs_per_stim = 1 if len(self.frequencies.shape) == 1 else self.frequencies.shape[1]

        self.monitor_freq = monitor_freq
        self.stim_duration_s = stim_duration_s
        self.shuffle_freqs = shuffle_freqs
        self._random_seed = random_seed
        self.wait_before_blink = wait_before_blink
        self.color_blink = color_blink
        self.frames = self._get_frames_from_frequencies(self.frequencies, self.monitor_freq, self.stim_duration_s)
        self.frames_per_stim = int(self.frames.shape[1] / self.n_freqs_per_stim)
        self._frame_intervals = None

    def get_params(self):
        """Returns a dictionary containing all parameters of the paradigm. The results of this method should
        be returned by `present()`. Any information that a decoder needs must be returned by this method.

        :return: a dictionary mapping a string naming a given parameter to its value
        """
        params = {'frequencies': self.frequencies,
                  'n_stim': self.n_stim,
                  'monitor_freq': self.monitor_freq,
                  'stim_duration_s': self.stim_duration_s,
                  'shuffle_freqs': self.shuffle_freqs,
                  'wait_before_blink': self.wait_before_blink,
                  'trig_stim_start': self.trig_stim_start,
                  'trig_stim_end': self.trig_stim_end,
                  'trig_stim_new_rep': self.trig_stim_new_rep
                  }
        return params

    def _presentation_preparation(self, save_frame_intervals_path):
        # set all stim objects to the same color (in case this was not the case before presentation)
        if self.color_blink is not None:
            self._change_color(self.objects_stim, self.color_blink)

        # render all steady stimuli into one image stimulus
        screenshot = psychopy.visual.BufferImageStim(self.window, stim=self.objects_steady)

        # time to read the options
        if self.wait_before_blink is not None:
            self._draw_frame()
            core.wait(self.wait_before_blink)

        if save_frame_intervals_path is not None:
            self.window.recordFrameIntervals = True
            self._frame_intervals = []

        # stimulation
        self._send_trigger(start=True)
        self._draw_frame()

        return screenshot

    def _presentation_end(self, save_frame_intervals_path):
        # final frame show all
        self._draw_frame()
        self._send_trigger(start=False)

        if save_frame_intervals_path is not None:
            self.window.recordFrameIntervals = False
            self.window.frameIntervals = []
            self.window.frameClock.reset()
            for i_fi, fi in enumerate(self._frame_intervals):
                plt.plot(fi, label=i_fi)
            plt.axhline(1 / self.monitor_freq, color='red', label='intended')
            plt.savefig(save_frame_intervals_path + ".png")

            with open(save_frame_intervals_path + ".pkl", 'wb') as f:
                pickle.dump(np.array(self._frame_intervals), f)

    def present(self, repetitions=None, save_frame_intervals_path=None):
        if repetitions is None:
            repetitions = 1  # todo

        if self.shuffle_freqs:
            self._shuffle_frequencies()

        screenshot = self._presentation_preparation(save_frame_intervals_path=save_frame_intervals_path)

        for frame_nr in range(self.frames.shape[-1]):
            # send new repetition trigger
            if frame_nr % self.frames_per_stim == 0:
                send_trig_if_parport(trigger=self.trig_stim_new_rep, parport=self.parport)

            # draw steady and stimulus objects
            screenshot.draw()
            self._draw_stim(frame_states=self.frames[:, frame_nr])
            self.window.flip()

            if save_frame_intervals_path is not None:
                # frame intervals per repetition
                if frame_nr % self.frames_per_stim == self.frames_per_stim-1:
                    self._frame_intervals.append(self.window.frameIntervals)
                    self.window.frameIntervals = []
                    self.window.frameClock.reset()

        self._presentation_end(save_frame_intervals_path=save_frame_intervals_path)

        return self.get_params()

    def _shuffle_frequencies(self) -> None:
        np.random.seed(self._random_seed)
        self._random_seed += 1
        shuffled = np.arange(self.n_stim)
        np.shuffle(shuffled)
        self.frequencies = self.frequencies[shuffled]

