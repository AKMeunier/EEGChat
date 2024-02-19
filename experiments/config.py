from psychopy import parallel
import numpy as np

from util.experiments.paradigms import SSVEP, ClickSelection, Oddball, CVEP, CVEPContinuous
from util.eeg.online import CVEPDecoder, ShamDecoder, SSVEPDecoder, CVEPContinuousDecoder, start_lsl_stream
from util.helper import possible_freqs


def config(info, experiment=True):
    # initialize some values, don't change!
    info['run_calibration'] = False
    info['calibration_reps'] = 0
    info['continuous_decoder'] = False
    decoder_class_continuous = None
    decoder_continuous_kwargs = None
    info['shared_dict_name'] = "shared_dict_0"
    debug_options = {"debug_question": False,
                     "skip_resting_state": False,
                     "skip_intro": False,
                     "skip_calibration": False}

    ###### CHANGE SETTINGS HERE ######

    info['fs_eeg'] = 1000
    info['n_channels_eeg'] = 6
    info['screen_freq'] = 60
    info['max_trials_per_block'] = 30
    info['flicker_box'] = True
    info['window_units'] = 'norm'
    info['include_history'] = True

    info['timing'] = {'rs_eyes_open': 180,  # lengths in seconds
                      'rs_eyes_closed': 180,
                      'question_rec': 8,
                      'wait_before_stim': 7
                      }

    debug = False  # If true, debug options below will override some settings

    if debug:
        # set lsl_inlet and parport to None to run without eeg connection
        lsl_inlet = None
        parport = None

        #lsl_inlet = start_lsl_stream()
        #parport = parallel.ParallelPort(address=0x0378)

        debug_options = {"debug_question": False,  # instead of recording a question always ask a standard question
                         "skip_resting_state": True,
                         "skip_intro": False,  # this may cause problems when using the continuous decoder, so set it to False then.
                         "skip_calibration": True}
        info['timing']['rs_eyes_open'] = 2
        info['timing']['rs_eyes_closed'] = 2
        info['timing']['question_rec'] = 8
        info['timing']['wait_before_stim'] = 1
        info['max_trials_per_block'] = np.inf

    else:
        lsl_inlet = start_lsl_stream()
        parport = parallel.ParallelPort(address=0x0378)  # set to None to run without parport

    info['parallel_port'] = parport

    info['trigger'] = {'experiment_start': 100,
                       'rs_open_start': 200,
                       'rs_open_end': 201,
                       'rs_closed_start': 210,
                       'rs_closed_end': 211,
                       'calibration_block_start': 110,
                       'calibration_block_end': 111,
                       'training_start': 120,
                       'new_block_train': 129,
                       'training_end': 121,
                       'evaluation_start': 130,
                       'new_block_eval': 139,
                       'evaluation_end': 131,
                       'accucary_eval_block_start': 140,
                       'accucary_eval_block_end': 141,
                       'experiment_end': 101,
                       'trial_start': 1,
                       'rec_audio_start': 2,
                       'rec_audio_end': 3,
                       'stim_start': 4,
                       'stim_end': 5,
                       'stim_new_rep': 6,
                       'audio_start': 7,
                       'audio_end': 8,
                       'trial_end': 9,
                       }


    # uncomment the paradigm you want to use, change settings in if statement below
    paradigm_switch = 'CVEP-CNN'
    #paradigm_switch = 'CVEP'
    #paradigm_switch = 'SSVEP'
    #paradigm_switch = 'ODDBALL'  # need to update, currently not working


    ##### Continuous paradigms #####
    if paradigm_switch == 'CVEP-CNN':
        info['continuous_decoder'] = True
        paradigm_train = [CVEPContinuous, ClickSelection]
        paradigm_eval = [CVEPContinuous]
        decoder_train = ShamDecoder()  # gets info from ClickSelection paradigm
        decoder_eval = ShamDecoder()   # gets info from CVEPContinuous paradigm
        kwarg_cvep_cont = {'delay_order': [0, 5, 7, 2, 9, 1, 6, 4, 8, 3],  # [0, 2, 4, 1, 3, 5, 6, 7, 8, 9],
                           'len_seq': 31,
                           'frame_shift': 9,
                           'wait_before_blink': info['timing']['wait_before_stim'],
                           'frame_nr_multiplier': 3,
                           'trig_stim_start': info['trigger']['stim_start'],
                           'trig_stim_end': info['trigger']['stim_end'],
                           'trig_stim_new_rep': info['trigger']['stim_new_rep'],
                           'max_repetitions': 7,
                           'color_blink': (0.5, 0.5, 0.5),
                           'shared_dict_name': info['shared_dict_name']
                           }
        kwarg_click = {}
        info['paradigm_kwargs_train'] = [kwarg_cvep_cont, kwarg_click]
        info['paradigm_kwargs_eval'] = [kwarg_cvep_cont]
        decoder_class_continuous = CVEPContinuousDecoder
        decoder_continuous_kwargs = {'fs_eeg': info['fs_eeg'],
                                     'screen_freq': info['screen_freq'],
                                     'n_eeg_channels': info['n_channels_eeg'],
                                     'eeg_window_len_ms': 250,
                                     'n_stim_min': 15,
                                     'classification_thresh': (0.6, 0.8),
                                     'trigger_channel': -1
                                     }
        info['run_calibration'] = False


    ##### Asynchronous paradigms #####
    elif paradigm_switch == 'CVEP':
        paradigm_train = [CVEP, ClickSelection]
        paradigm_eval = [CVEP]
        decoder_train = ShamDecoder()
        decoder_eval = CVEPDecoder(lsl_inlet=lsl_inlet, fs_eeg=info['fs_eeg'], screen_freq=info['screen_freq'],
                                   shift_crosscorr=1)
        kwarg_cvep = {'delay_order': [0,5,7,2,9,1,6,4,8,3],#[0, 2, 4, 1, 3, 5, 6, 7, 8, 9],
                      'len_seq': 31,
                      'frame_shift': 9,
                      'wait_before_blink': info['timing']['wait_before_stim'],
                      'frame_nr_multiplier': 3,
                      'trig_stim_start': info['trigger']['stim_start'],
                      'trig_stim_end': info['trigger']['stim_end'],
                      'trig_stim_new_rep': info['trigger']['stim_new_rep'],
                      'repetitions': 1,
                      'color_blink': (0.5, 0.5, 0.5)
                      }
        kwarg_click = {}
        info['paradigm_kwargs_train'] = [kwarg_cvep, kwarg_click]
        info['paradigm_kwargs_eval'] = [kwarg_cvep]
        info['run_calibration'] = True
        info['calibration_reps'] = 20

    elif paradigm_switch == 'SSVEP':
        paradigm_train = [SSVEP, ClickSelection]  # must be list of paradigms
        paradigm_eval = [SSVEP]
        decoder_train = ShamDecoder()
        decoder_eval = SSVEPDecoder(lsl_inlet=lsl_inlet, fs_eeg=info['fs_eeg'], bandwidth=1.)
        if info["screen_freq"] == 75:
            freqs = np.array([7.5, 12.5, 25.])
        elif info["screen_freq"] == 60:
            freqs = np.array([6., 10., 15.])
        else:
            raise AttributeError(f"Need to define SSVEP frequencies for the monitor frequency {info['screen_freq']} Hz. "
                                 f"Possible frequencies:\n{possible_freqs(info['screen_freq'])}")
        ssvep_order = [[0, 1, 2],   # KW 1
                       [1, 2, 0],   # KW 2
                       [2, 0, 1],   # KW 3
                       [2, 1, 0],   # KW 4
                       [0, 2, 1],   # KW 5
                       [1, 0, 2],   # KW 6
                       [1, 1, 1],   # Correction
                       [0, 2, 0],   # More
                       [0, 0, 0],   # None
                       [2, 2, 2]]   # Finished
        ssvep_freqs = np.array([freqs[so] for so in ssvep_order])
        kwarg_ssvep = {'frequencies': ssvep_freqs,
                       'color_blink': (0.5, 0.5, 0.5), #(1., 1., 1.),
                       'wait_before_blink': info['timing']['wait_before_stim'],
                       'trig_stim_start': info['trigger']['stim_start'],
                       'trig_stim_end': info['trigger']['stim_end'],
                       'trig_stim_new_rep': info['trigger']['stim_new_rep'],
                       'monitor_freq': info['screen_freq'],
                       'stim_duration_s': 1.,
                       'shuffle_freqs': False,
                       'random_seed': 1234
                       }
        kwarg_click = {}
        info['paradigm_kwargs_train'] = [kwarg_ssvep, kwarg_click]
        info['paradigm_kwargs_eval'] = [kwarg_ssvep]

    elif paradigm_switch == 'ODDBALL':
        raise DeprecationWarning('Oddball paradigm needs to be updated before use!')

        # paradigm_train = [Oddball, ClickSelection]  # must be list of paradigms
        # paradigm_eval = [Oddball]  # must be list
        # kwarg_oddball = {'rep_per_object': 4,
        #                  'simultaneous': 2,
        #                  'blink_duration': 0.05,
        #                  'blink_distance': 0.15,
        #                  'random_seed': 1234,
        #                  'color_blink_on': (1., 1., 1.),
        #                  'color_blink_off': (0.5, 0.5, 0.5),
        #                  'wait_before_blink': info['timing']['wait_before_stim'],
        #                  'trig_stim_start': info['trigger']['stim_start'],
        #                  'trig_stim_end': info['trigger']['stim_end']
        #                  }
        # kwarg_click = {}
        # info['paradigm_kwargs_train'] = [kwarg_oddball, kwarg_click]
        # info['paradigm_kwargs_eval'] = [kwarg_oddball]
        # info['flicker_box'] = False  # does not work with true for oddball yet


    info['texts'] = {'instruction_start': 'In this experiment you will be answering questions by selecting short '
                                          'answers on a screen. In the beginning of the experiment, you will select '
                                          'these answers by clicking on them. Later you will select the answers '
                                          'simply by looking at the words. It will be inferred from your brain '
                                          'activity, which answer your are focussing on.\n\nWe will now begin the '
                                          'experiment with a relaxation period. You will later receive more detailed '
                                          'instructions on your tasks during each round.',


                     'instruction_begin_training': 'We are now beginning a training round. You will be shown a '
                                                   'description of a scenario. Pretend that you are on a '
                                                   'phone call with the experimenter, trying to fulfill this given '
                                                   'scenario. The experimenter does not know the details of what you '
                                                   'want to achieve and will ask you questions.\n\nYou can '
                                                   'choose from presented one-word answers by first focussing on your '
                                                   'choice during a flickering phase, and afterwards clicking it. '
                                                   'Please do not speak during the experiment, only use the presented '
                                                   'keywords.\n\nOn the following pages you will see a detailed '
                                                   'explanation of all the steps. Take your time to familiarize '
                                                   'yourself with the task and feel free to ask any questions you may '
                                                   'have.',

                     'instruction_task_descriptions': ["At the beginning you will be shown this screen. Please listen "
                                                       "and wait while the experimenter asks their question.",
                                                       "The question will then be transcribed automatically and shown "
                                                       "at the top. Furthermore, a language model will automatically "
                                                       "generate six possible short answers. Please read those answers "
                                                       "and choose one that best fits with your objective.\n\nFocus your "
                                                       "attention on your choice. After a few seconds the boxes will "
                                                       "start flickering for a few seconds. Please keep your attention "
                                                       "on your choice, and try not to blink or swallow.\n\nAfter the "
                                                       "flickering has stopped, use the mouse to click on your choice.",
                                                       "After you clicked your choice, another language model will "
                                                       "translate your choice into a full sentence answer. This answer "
                                                       "will be read out loud by a computer voice. After that, the "
                                                       "experimenter will ask another question.",
                                                       "In addition to the generated answers you have four other "
                                                       "options on the right side of the screen. If none of the six "
                                                       "generated words present a reasonable answer, please choose one "
                                                       "of these:\n\n\n"
                                                       "Correction: The computer voice will answer 'I am sorry, I "
                                                       "misspoke earlier' and the experimenter will ask about what you "
                                                       "want to correct.\n\n"
                                                       "More / Previous: Show six more keyword options / Show the "
                                                       "previous six keywords\n\n"
                                                       "None: The question cannot be answered with any of the twelve "
                                                       "keywords (six original and six after clicking 'More'). "
                                                       "This can for example be the case if there is no appropriate "
                                                       "keyword. The computer voice will answer 'I am sorry, I cannot "
                                                       "answer this question right now.'\n\n"
                                                       "Finished: You have achieved the scenario goal and want to hang "
                                                       "up the phone. The computer voice will say 'Thank you, goodbye'."
                                                       "\n\n\nAfter you choose \'Finished\', the "
                                                       "scenario ends and a new scenario will start.",
                                                       "If you cannot achieve the scenario after {} questions, the "
                                                       "scenario will end and the next one will start.\n\nIf you have "
                                                       "any more questions about the experiment, please ask them "
                                                       "now.\n\nIf you are ready, press space to "
                                                       "continue.".format(info['max_trials_per_block'])
                                                       ],

                     'instruction_task_descriptions_imgs': ["figs/explanation_1.png", "figs/explanation_3.png",
                                                            "figs/explanation_5.png", "figs/explanation_6.png",
                                                            "figs/explanation_3.png"],

                     'instruction_begin_eval': 'We are now beginning the experiment scenarios. You will be shown a '
                                               'series of scenarios, similar to the training scenario. Again, pretend '
                                               'that you are on a phone call with the experimenter, trying to fulfill '
                                               'this given scenario. During these scenarios, the answer will be '
                                               'determined from your brain activity while you focus on your choice. '
                                               'You do not need to click.\n\nIf an answer is chosen that you did not '
                                               'intend to choose, please still try to fulfill the given scenario, e.g. '
                                               'by selecting "Correction" and answering a question again.\n\nIf you '
                                               'have any more questions about the experiment, please ask them now.',
                     'instruction_begin_calibration': 'We are now beginning a calibration phase. You will be shown a '
                                                      'question and ten different one-word answers. From these '
                                                      'answers, please choose the correct one. Focus your attention on '
                                                      'this answer, while the answers flicker for several seconds. '
                                                      'After the flickering has stopped, use your mouse to click on '
                                                      'the chosen answer.',
                     'instruction_begin_acceval': "We are now starting an evaluation of how good the system is at "
                                                  "identifying the keyword you are focussing on. You will be shown "
                                                  "simple questions, as well as short answers. Please focus on the "
                                                  "correct answer during the flickering phase."}

    knowledge_base = {'NAME': ["Anna", "Mayer", "Anna Mayer", "David Mayer", "Laura", "Oliver",
                               "Peter", "Sophia", "Tim", "Marianne", "Maria", "Felix"],
                      'ADDRESS': ["15 Flowerstreet", "7 Southroad", "104 Mainstreet", "28 Bumblebee Lane",
                                  "56 Park Avenue", "7 St. Michael Road", "4 Hartford Road", "9 Oak Lane",
                                  "101 First Avenue", "12 Club Road", "123 Magnolia Ring", "1 Walterstreet", ]}

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    scenarios_eval = [["MUSEUM", "Ask what the opening hours of the museum are."],
                       ["PIZZERIA", "Make a reservation for {n_people} people at the pizzeria for {day} at {time} under the "
                             "name '{name}'.".format(n_people=np.random.choice(a=[2, 3, 4, 5, 6]),
                                                     day=np.random.choice(a=days),
                                                     time=np.random.choice(a=['1pm', '2pm', '6pm', '7pm', '8pm']),
                                                     name=np.random.choice(a=["Anna", "Mayer", "David Mayer", "Oliver"]))],
                     ["FITNESS", "Call your fitness studio and schedule a session with your trainer {name} for {day} "
                                 "{timeofday} under the name {yourname}.".format(day=np.random.choice(a=days[:5]),
                                                                                 timeofday=np.random.choice(a=['morning', 'afternoon']),
                                                                                 name=np.random.choice(a=["Peter", "Sophia", "Maria", "Felix"]),
                                                                                 yourname=np.random.choice(a=["Anna Mayer", "David Mayer"]))],
                     ["DOCTOR", "Make a doctor's appointment for next {day} {timeofday} under the name "
                                     "'{name}'.".format(name=np.random.choice(a=["David Mayer", "Anna Mayer"]),
                                                        timeofday=np.random.choice(a=['morning', 'afternoon']),
                                                        day=np.random.choice(a=days[:5]))],
                     ["CAFE", "Make a reservation at the cafe for {n_people} people next {day} at {time} under the name "
                             "{name}.".format(n_people=np.random.choice(a=[2, 3, 4, 5, 6]),
                                              day=np.random.choice(a=days),
                                              time=np.random.choice(a=['2pm', '3pm', '4pm', '5pm']),
                                              name=np.random.choice(a=["Anna", "Laura", "Oliver", "Tim"]))],
                     ["TAXI", "Order a taxi to {address} for {day} at {time} for {n_people} "
                             "people.".format(n_people=np.random.choice(a=[2, 5]),
                                              day=np.random.choice(a=days),
                                              time=np.random.choice(a=['9am', '11am', '3pm', '5pm']),
                                              address=np.random.choice(a=["15 Flowerstreet", "7 Southroad",
                                                                          "104 Mainstreet", "28 Bumblebee Lane"]))],
                     ["HAIRDRESSER", "Make an appointment at the hair salon to have your hair {job} on {day} {timeofday}"
                                    " under the name {name}.".format(job=np.random.choice(a=["cut", "colored", "styled"]),
                                                                     day=np.random.choice(a=days[:5]),
                                                                     timeofday=np.random.choice(a=['morning', 'afternoon']),
                                                                     name=np.random.choice(a=["Anna", "Laura", "Oliver", "Tim"]))]
                 ]

    scenarios_train = [["TRAINING", "Training scenario. Please follow the experimenter's instructions."]]

    text_calibration = {'questions': ['What color is an elephant?',
                                      'How many legs does a dog have?',
                                      "Please select 'More' now.",
                                      'What do plants need to survive?',
                                      "Please select 'None' now.",
                                      'What continent are we on right now?',
                                      'What are trees made out of?',
                                      'How many fingers do humans typically have in total?',
                                      "Please select 'Correction' now.",
                                      'Where do kangaroos live?',
                                      "Please select 'Finished' now.",
                                      "Which city are we in right now?",
                                      "Which of these animals can fly?",
                                      "Please select 'None' now.",
                                      "Please select 'More' now.",
                                      "What color is grass?",
                                      "Please select 'Finished' now.",
                                      "Which of these animals lives under water?",
                                      "Please select 'Correction' now.",
                                      "What planet are we living on?"
                                      ],
                        'keywords': [['Orange', 'Grey', 'Yellow', 'Green', 'Red', 'Pink'],
                                     ['One', 'Two', 'Three', 'Four', 'Five', 'Six'],
                                     ['Elefant', 'Mouse', 'Cat', 'Dog', 'Bird', 'Lion'],
                                     ['Friends', 'Coffee', 'Water', 'Darkness', 'Salt', 'Soup'],
                                     ['Yes', 'Never', 'Maybe', 'No', 'Sometimes', 'Always'],
                                     ['Africa', 'South America', 'North America', 'Asia', 'Australia', 'Europe'],
                                     ['Wood', 'Aluminium', 'Pudding', 'Glass', 'Iron', 'Potatoes'],
                                     ['Four', 'Fifteen', 'Eleven', 'One', 'Ten', 'Three'],
                                     ['Happy', 'Sad', 'Ecstatic', 'Tired', 'Hungry', 'Good'],
                                     ['Sweden', 'Australia', 'Mars', 'Antarctica', 'Narnia', 'Brazil'],
                                     ['Pizza', 'Sushi', 'Burgers', 'Tacos', 'Risotto', 'Pad Thai'],
                                     ['London', 'Hong Kong', 'Vienna', 'Bogota', 'Chicago', 'Lagos'],
                                     ['Bird', 'Elephant', 'Dog', 'Mouse', 'Snake', 'Bear'],
                                     ['Yes', 'Never', 'Maybe', 'No', 'Sometimes', 'Always'],
                                     ['One', 'Two', 'Three', 'Four', 'Five', 'Six'],
                                     ['Orange', 'Grey', 'Yellow', 'Green', 'Red', 'Pink'],
                                     ['Friends', 'Coffee', 'Water', 'Darkness', 'Salt', 'Soup'],
                                     ['Elephant', 'Bear', 'Bird', 'Mouse', 'Fish', 'Cat'],
                                     ['Pizza', 'Sushi', 'Burgers', 'Tacos', 'Risotto', 'Pad Thai'],
                                     ['Mars', 'Jupyter', 'Venus', 'Mercury', 'Saturn', 'Earth'],

                                     ],
                        'idx_correct': [1, 3, 7, 2, 8, 5, 0, 4, 6, 1, 9,
                                        2, 0, 8, 7, 3, 9, 4, 6, 5]}



    ###### END SETTINGS ######

    info['texts']['scenarios_train'] = scenarios_train
    info['texts']['scenarios_eval'] = scenarios_eval
    info['texts']['text_calibration'] = text_calibration
    info['texts']['knowledge_base'] = knowledge_base

    if experiment:
        return info, paradigm_train, paradigm_eval, decoder_train, decoder_eval, parport, lsl_inlet, debug_options
    else:
        return info, decoder_class_continuous, decoder_continuous_kwargs, lsl_inlet

