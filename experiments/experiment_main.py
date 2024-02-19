import warnings
import numpy as np
from psychopy import visual, core, event, gui
from shared_memory_dict import SharedMemoryDict

from util.experiments.standard import resting_state, change_bg_color, check_escape, send_trig_if_parport, paginated_text_with_pics
from util.chatbot import generate_answer_finetuned as generate_answer
from util.chatbot import generate_keywords_details as generate_keywords
from util.audio import wav_to_text, text_to_wav, record, play_wav
from util.helper import create_folder_if_not_exist, datestr, save_json
from config import config


class Experiment:
    def __init__(self, window, info, paradigm_train, paradigm_eval, decoder_train,
                 decoder_eval, run_calibration, parport, debug_options, flicker_box=False):
        self.info = info
        if self.info['continuous_decoder']:
            self.shared_dict = SharedMemoryDict(name=info['shared_dict_name'], size=1024 * 2)
            print("Shared dict opened.")
            try:
                if self.shared_dict["decoder_started"]:
                    print("Running decoder script detected.")
                else:
                    raise ConnectionError("This should never happen? :(")
            except KeyError:
                raise ConnectionError("Start the decoder_main.py script first.")

        self.debug_options = debug_options
        self.info_trials = {}
        self.scenarios_train = info['texts']['scenarios_train']
        self.scenarios_eval = info['texts']['scenarios_eval']
        self.decoder_train = decoder_train
        self.decoder_eval = decoder_eval
        self.calibration = run_calibration and not self.debug_options["skip_calibration"]
        self.parport = parport
        self.flicker_box = flicker_box

        self.info['paradigms'] = {'train': [p.__name__ for p in paradigm_train],
                                  'eval': [p.__name__ for p in paradigm_eval]}

        # create result folders
        self.info['filename'] = self.info['participant'].replace(' ', '-') + "_" + datestr()
        self.info['result_dir'] = "../data/experiments/{}/".format(info['filename'])
        create_folder_if_not_exist(self.info['result_dir'])
        create_folder_if_not_exist(self.info['result_dir'] + 'audio/')
        create_folder_if_not_exist(self.info['result_dir'] + 'frame_intervals/')
        self.info_trials_filename = self.info['result_dir'] + self.info['filename'] + "_trials.json"

        if self.info['continuous_decoder']:
            self.shared_dict["result_dir"] = self.info['result_dir']

        # write experiment info file
        save_json(dictionary=info, file_name=self.info['result_dir'] + self.info['filename'] + "_info.json")

        # trial information
        self.block_nr = 0
        self.trial_nr = 0
        self.block_clock = core.Clock()
        self.n_blocks_calibrate = 1 * self.calibration
        self.n_blocks_train = len(self.scenarios_train)
        self.n_blocks = len(self.scenarios_eval)
        self._new_question = True
        self._new_keywords = True
        self._end_block = False
        self._permanent_kw_options1 = ['Correction', 'More', 'None', 'Finished']
        self._permanent_kw_options2 = ['Correction', 'Previous', 'None', 'Finished']
        self._keywords = [''] * 12
        self._history = ""

        if self.calibration:
            assert len(self.info['texts']['text_calibration']['questions']) == len(self.info['texts']['text_calibration']['keywords'])
            assert len(self.info['texts']['text_calibration']['questions']) == len(self.info['texts']['text_calibration']['idx_correct'])

        # create screen layout
        self.window = window

        units = window.units

        if units == "height":
            # (width, height)
            center = (0, -0.05,)
            dist = (0.4, 0.25)
            factor = (1., 0.9)
            boxsize = (0.35, 0.2)
            letter_height_factor = 1.

        elif units == "norm":
            # (width, height)
            center = (0, -0.05,)
            dist = (0.4, 0.25)
            factor = (1.25, 0.9 * 2)
            boxsize = (0.35, 0.2)
            letter_height_factor = 2.

        else:
            raise NotImplementedError("Attribute 'units' in psychopy.visual.Window must be 'height' or 'norm', or "
                                      "layout must be defined for different value.")

        # define layout of boxes
        boxes_pos = (np.array([(-1.5, 0.5), (-0.5, 0.5), (0.5, 0.5),
                               (-1.5, -0.5), (-0.5, -0.5), (0.5, -0.5),
                               (1.5, 1.5), (1.5, 0.5), (1.5, -0.5), (1.5, -1.5)])
                     * np.array(dist) + np.array(center)) * np.array(factor)
        boxes_size = np.array([boxsize for i in range(len(boxes_pos))]) * np.array(factor)

        question_pos = (np.array((-0.5, 1.5)) * np.array(dist) + np.array(center)) * np.array(factor)
        question_size = (boxes_pos[2][0] - boxes_pos[0][0] + boxes_size[0][0], boxes_size[0][1])

        answer_pos = (np.array((-0.5, -1.5)) * np.array(dist) + np.array(center)) * np.array(factor)
        answer_size = question_size

        scenario_pos = (np.array((0., 2.2)) * np.array(dist) + np.array(center)) * np.array(factor)
        scenario_size = (boxes_pos[-1][0] - boxes_pos[0][0] + boxes_size[0][0], boxes_size[0][1] / 2)

        kw_letter_height = 0.04 * letter_height_factor
        question_letter_height = 0.03 * letter_height_factor
        answer_letter_height = 0.03 * letter_height_factor
        scenario_letter_height = 0.03 * letter_height_factor
        self._instructions_letter_height = 0.02 * letter_height_factor
        self._scenario_intro_letter_height = 0.04 * letter_height_factor

        self.text_question = visual.TextBox2(win=self.window,
                                             text='',
                                             lineBreaking='uax14',
                                             letterHeight=question_letter_height,
                                             pos=question_pos,
                                             size=question_size,
                                             padding=(0.05, 0.),
                                             borderWidth=1,
                                             borderColor=(0.5, 0.5, 0.5),
                                             borderColorSpace='rgb',
                                             color=1.,
                                             colorSpace='rgb',
                                             fillColor=None,
                                             fillColorSpace='rgb',)

        self.text_answer = visual.TextBox2(win=self.window,
                                           text='',
                                           lineBreaking='uax14',
                                           letterHeight=answer_letter_height,
                                           pos=answer_pos,
                                           size=answer_size,
                                           padding=(0.05, 0.),
                                           borderWidth=1,
                                           borderColor=(0.5, 0.5, 0.5),
                                           borderColorSpace='rgb',
                                           color=(1., 1., 1.),
                                           colorSpace='rgb',
                                           fillColor=None,
                                           fillColorSpace='rgb',)

        self.text_scenario = visual.TextBox2(win=self.window,
                                             text='Calibration: Focus on the correct answer, please!',
                                             lineBreaking='uax14',
                                             alignment='center',
                                             letterHeight=scenario_letter_height,
                                             pos=scenario_pos,
                                             size=scenario_size,
                                             padding=(0.05, 0.),
                                             borderColorSpace='rgb',
                                             color=(0.05, 0.05, 0.05),
                                             colorSpace='rgb',
                                             fillColor=None,
                                             fillColorSpace='rgb')

        # create boxes and text
        self.boxes_border = []
        self.boxes_texts = []

        for i in range(10):
            t = visual.TextBox2(win=self.window,
                                text='',
                                alignment='center',
                                lineBreaking='uax14',
                                pos=boxes_pos[i],
                                size=(boxes_size[i][0] * 0.95, boxes_size[i][1] * 0.95),
                                letterHeight=kw_letter_height,
                                color=(1., 1., 1.),
                                colorSpace='rgb',
                                fillColor=None,
                                fillColorSpace='rgb',
                                borderColorSpace='rgb')
            self.boxes_texts.append(t)

            bb = visual.TextBox2(win=self.window,
                                 text='',
                                 pos=boxes_pos[i],
                                 size=boxes_size[i],
                                 borderWidth=1,
                                 borderColor=(0.5, 0.5, 0.5),
                                 borderColorSpace='rgb',
                                 colorSpace='rgb',
                                 fillColor=None,
                                 fillColorSpace='rgb',)
            self.boxes_border.append(bb)

        self.n_boxes = len(self.boxes_texts)

        if self.flicker_box:
            self.boxes_overlay = visual.ElementArrayStim(win=self.window,
                                                         units=units,
                                                         fieldPos=(0., 0.),
                                                         fieldSize=self.window.size,
                                                         fieldShape="sqr",
                                                         nElements=self.n_boxes,
                                                         elementTex=None,
                                                         elementMask=np.ones((256, 256)),
                                                         xys=boxes_pos,
                                                         sizes=boxes_size,
                                                         opacities=0.5
                                                         )

            objects_stim = self.boxes_overlay
            objects_steady = self.boxes_border + self.boxes_texts + [self.text_question, self.text_scenario]
        else:
            warnings.warn('Flickering words can be slower than flickering boxes.')
            objects_stim = self.boxes_texts
            objects_steady = self.boxes_border + [self.text_question, self.text_scenario]

        # instantiate paradigms
        self.paradigm_train = [paradigm_train[p](window=self.window,
                                                 objects_stim=objects_stim if "ClickSelection" not in paradigm_train[p].__name__ else self.boxes_border,
                                                 objects_steady=objects_steady,
                                                 parport=self.info['parallel_port'],
                                                 show_obj_stim_when_waiting=not self.flicker_box,
                                                 **self.info['paradigm_kwargs_train'][p])
                               for p in range(len(paradigm_train))]

        self.paradigm_eval = [paradigm_eval[p](window=self.window,
                                               objects_stim=objects_stim if "ClickSelection" not in paradigm_train[p].__name__ else self.boxes_border,
                                               objects_steady=objects_steady,
                                               parport=self.info['parallel_port'],
                                               show_obj_stim_when_waiting=not self.flicker_box,
                                               **self.info['paradigm_kwargs_eval'][p])
                              for p in range(len(paradigm_eval))]

        send_trig_if_parport(trigger=self.info['trigger']['experiment_start'], parport=self.parport)

    def set_question(self, text):
        self.text_question.setText(text)

    def set_answer(self, text):
        self.text_answer.setText(text)

    def set_scenario(self, text):
        self.text_scenario.setText(text)

    def generate_keywords(self, question):
        self._keywords = generate_keywords(question=question,
                                           n=12,
                                           knowledge_base=self.info['texts']['knowledge_base'])
        return self._keywords

    def load_keywords(self, keywords):
        self._keywords = keywords

    def set_keywords(self, which):
        if which == 'first':
            list_text = self._keywords[:6] + self._permanent_kw_options1
            print(list_text)
        elif which == 'last':
            list_text = self._keywords[6:] + self._permanent_kw_options2
            print(list_text)
        elif which == 'empty':
            list_text = [''] * 6 + self._permanent_kw_options1
        else:
            raise ValueError("Argument 'which' must be in ['first', 'last', 'empty'].")
        for i, text in enumerate(list_text):
            self.boxes_texts[i].setText(text)
            self.boxes_texts[i].setColor((1., 1., 1.), colorSpace='rgb')
        return list_text

    def set_box_color(self, box_nr, color, colorSpace='rgb'):
        self.boxes_texts[box_nr].setColor(color, colorSpace=colorSpace)

    def draw(self, draw_answer=True):
        self.text_question.draw()
        self.text_scenario.draw()
        for i in range(len(self.boxes_border)):
            self.boxes_border[i].draw()
            self.boxes_texts[i].draw()
        if draw_answer:
            self.text_answer.draw()
        self.window.flip()

    def record_question(self, wav_file):
        question = None
        n_dots = 3
        while question is None:
            print('Recording question')
            self.set_question('Please wait'+'.' * n_dots)
            self.text_question.setColor('red', colorSpace='named')
            self.set_keywords(which='empty')
            self.draw(draw_answer=False)

            # self.window.getMovieFrame(buffer='front')
            # self.window.saveMovieFrames("screenshot1.png")

            send_trig_if_parport(trigger=self.info['trigger']['rec_audio_start'], parport=self.parport)
            record(wav_file=wav_file, duration=self.info['timing']['question_rec'])
            send_trig_if_parport(trigger=self.info['trigger']['rec_audio_end'], parport=self.parport)
            question = wav_to_text(wav_path=wav_file, language=self.info['language'])
            if self.debug_options["debug_question"]:
                question = "How are you today"
            check_escape(self.window)
            n_dots += 1
        question = question
        print(question)
        self.text_question.setColor((1., 1., 1.), colorSpace='rgb')
        self.set_question(question)
        self.draw(draw_answer=False)

        # self.window.getMovieFrame(buffer='front')
        # self.window.saveMovieFrames("screenshot2.png")

        return question

    def output_answer(self, answer, wav_file):
        print(answer)
        self.set_answer(answer)
        self.draw(draw_answer=True)
        send_trig_if_parport(trigger=self.info['trigger']['audio_start'], parport=self.parport)
        text_to_wav(text=answer, wav_file=wav_file)
        send_trig_if_parport(trigger=self.info['trigger']['audio_end'], parport=self.parport)
        play_wav(wav_file)

    def show_scenario_text(self, scenario_description, scenario_name):
        print("--- New scenario ---")
        print("Scenario: " + scenario_name)
        self.set_scenario('Scenario: ' + scenario_description)

        self.show_instructions(text='Scenario:\n\n' + scenario_description + '\n\n',
                               bg_color=(-1., -1., -1.),
                               alignment='center')

    def show_instructions(self, text, bg_color=(0.0, 0.0, 0.0), alignment='left'):
        instructions = visual.TextBox2(win=self.window,
                                       text=text + '\n\nWhen you are ready, press space to continue.',
                                       letterHeight=self._instructions_letter_height,
                                       color=(1., 1., 1.),
                                       colorSpace='rgb',
                                       pos=(0., 0.),
                                       alignment=alignment,
                                       lineBreaking='uax14')

        # instructions
        change_bg_color(self.window, bg_color)
        instructions.draw()

        self.window.flip()
        print("Waiting for keypress...")
        pressed = event.waitKeys(keyList=['space', 'escape'])
        if pressed[-1] == 'space':
            event.clearEvents()
        elif pressed[-1] == 'escape':
            window.close()
            core.quit()

    def trial(self, training=False):
        send_trig_if_parport(trigger=self.info['trigger']['trial_start'], parport=self.parport)
        self._end_block = False

        self.info_trials[self.block_nr]['trials'][self.trial_nr] = {}
        # set question
        if self._new_question:
            wav_file = self.info['result_dir'] + 'audio/' + self.info['filename'] + \
                       f"_block{self.block_nr}_trial{self.trial_nr}_question.wav"
            question = self.record_question(wav_file)
            self.info_trials[self.block_nr]['trials'][self.trial_nr]['question_audio_file'] = wav_file

        else:
            question = self.info_trials[self.block_nr]['trials'][self.trial_nr - 1]['question']
            self.info_trials[self.block_nr]['trials'][self.trial_nr]['question_audio_file'] = ''
            self.set_question(question)
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['question'] = question

        # generate keywords with GPT
        if self._new_keywords:
            self.generate_keywords(question)
            keywords_show = self.set_keywords(which='first')
        else:  # selected "More" / "Previous"
            if self.info_trials[self.block_nr]['trials'][self.trial_nr - 1]['keywords'][7] == self._permanent_kw_options2[1]:
                keywords_show = self.set_keywords(which='first')
            else:
                keywords_show = self.set_keywords(which='last')
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['keywords'] = keywords_show

        if training:
            paradigms = self.paradigm_train
            decoder = self.decoder_train
            block_type = 'train'
        else:
            paradigms = self.paradigm_eval
            decoder = self.decoder_eval
            block_type = 'eval'

        # clear decoder data and lsl buffer
        decoder.clear_data()

        # present stimuli
        params = []
        for p_nr, paradigm in enumerate(paradigms):
            params.append(paradigm.present(save_frame_intervals_path=None if p_nr != 0 else self.info['result_dir'] +
                                                                                            "frame_intervals/" +
                                                                                            self.info['filename'] +
                                                                                            f"_block{self.block_nr}"
                                                                                            f"_trial{self.trial_nr}"
                                                                                            f"_frame_intervals"))
            self.info_trials[self.block_nr]['trials'][self.trial_nr]['params_stim_' + self.info['paradigms'][block_type][p_nr]] = params[-1]
            check_escape(self.window)

        # self.window.getMovieFrame(buffer='front')
        # self.window.saveMovieFrames("screenshot3.png")

        # decode
        if block_type == 'train':
            print(f"Eval decoder: {self.decoder_eval.decode(params[0], pull_new_data=True)}")

        choice = decoder.decode(params[-1], pull_new_data=True)
        print(choice)
        # Set to "cannot answer right now" if the decoder returns no result
        if choice is None:
            choice = 8

        if block_type == 'train':
            print(f"Clicked: {choice}")
            if info['continuous_decoder']:
                self.shared_dict['clicked'] = choice
            self.decoder_eval.fit_training(choice=choice)

        self.info_trials[self.block_nr]['trials'][self.trial_nr]['kw_choice'] = choice
        check_escape(self.window)

        # highlight chosen box
        self.set_box_color(box_nr=choice, color='blue', colorSpace='named')
        self.draw(draw_answer=False)

        # self.window.getMovieFrame(buffer='front')
        # self.window.saveMovieFrames("screenshot4.png")

        wav_file_answer = self.info['result_dir'] + 'audio/' + self.info['filename'] + f"_block{self.block_nr}_trial{self.trial_nr}_answer.wav"
        if choice == 6:
            answer = "I am sorry, I misspoke earlier."
            self._history += "\nQuestion: " + question + "\nAnswer: " + answer
            self._new_question = True
            self._new_keywords = True
            self.output_answer(answer=answer, wav_file=wav_file_answer)
        elif choice == 7:
            # not adding to history because it is still the same question
            self._new_keywords = False
            self._new_question = False
            answer = ''
            wav_file_answer = ''
        elif choice == 8:
            # not adding to history because question could not be answered
            answer = "I am sorry, I cannot answer this question right now."
            self._new_keywords = True
            self._new_question = True
            self.output_answer(answer=answer, wav_file=wav_file_answer)
        elif choice == 9:
            answer = "Thank you, good bye."
            self._history += "\nQuestion: " + question + "\nAnswer: " + answer
            self.output_answer(answer=answer, wav_file=wav_file_answer)
            self.set_question(" ")
            self._end_block = True
        else:
            # generate full sentence answer from keyword with gpt
            if self.info['include_history']:
                answer = generate_answer(question=question, keyword=keywords_show[choice], history=self._history + "\n")
            else:
                answer = generate_answer(question=question, keyword=keywords_show[choice])
            self._history += "\nQuestion: " + question + "\nAnswer: " + answer
            self._new_keywords = True
            self._new_question = True
            self.output_answer(answer=answer, wav_file=wav_file_answer)
            # self.window.getMovieFrame(buffer='front')
            # self.window.saveMovieFrames("screenshot5.png")

        self.info_trials[self.block_nr]['trials'][self.trial_nr]['full_answer'] = answer
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['answer_audio_file'] = wav_file_answer
        self.trial_nr += 1
        send_trig_if_parport(trigger=self.info['trigger']['trial_end'], parport=self.parport)

    def block(self, scenario, training=False):
        if training:
            send_trig_if_parport(trigger=self.info['trigger']['new_block_train'], parport=self.parport)
        else:
            send_trig_if_parport(trigger=self.info['trigger']['new_block_eval'], parport=self.parport)
        self.show_scenario_text(scenario_description=scenario[1], scenario_name=scenario[0])
        self.info_trials[self.block_nr] = {'scenario': scenario[1],
                                           'block_type': 'train' if training else 'eval',
                                           'trials': {}}
        change_bg_color(self.window, 'black')

        self.block_clock.reset()
        self.trial_nr = 0
        self._new_keywords = True
        self._new_question = True
        self._end_block = False
        self._history = ""
        while self.trial_nr < self.info['max_trials_per_block']:
            self.trial(training=training)
            if self._end_block:
                self.info_trials[self.block_nr]['duration'] = self.block_clock.getTime()
                save_json(dictionary=self.info_trials, file_name=self.info['result_dir'] + self.info['filename'] + "_trials.json")
                break
            else:
                save_json(dictionary=self.info_trials, file_name=self.info['result_dir'] + self.info['filename'] + "_trials.json")
        print(f"Block ended. Trials used: {self.trial_nr}")

    def calibration_block(self):
        print("--- Starting calibration round ---")
        questions = self.info['texts']['text_calibration']['questions']
        keywords = self.info['texts']['text_calibration']['keywords']
        idx_correct = self.info['texts']['text_calibration']['idx_correct']

        # instructions
        self.show_instructions(text=self.info['texts']['instruction_begin_calibration'])

        # clear decoder data and buffer
        self.decoder_eval.clear_data()

        # trigger
        send_trig_if_parport(trigger=self.info['trigger']['calibration_block_start'], parport=self.parport)

        self.info_trials[self.block_nr] = {'scenario': 'Calibration',
                                           'block_type': 'calibration',
                                           'trials': {}}

        change_bg_color(self.window, 'black')
        self.block_clock.reset()
        self.trial_nr = 0
        trials_correct = []
        for n_trial in range(len(questions)):
            # set question
            question = questions[n_trial]
            print(question)
            self.set_question(question)

            # set keywords
            self.load_keywords(keywords[n_trial] + [' '] * 6)
            keywords_show = self.set_keywords(which='first')

            repeat = True
            while repeat:
                send_trig_if_parport(trigger=self.info['trigger']['trial_start'], parport=self.parport)
                self.info_trials[self.block_nr]['trials'][self.trial_nr] = {}
                self.info_trials[self.block_nr]['trials'][self.trial_nr]['question'] = question
                self.info_trials[self.block_nr]['trials'][self.trial_nr]['keywords'] = keywords_show

                # present stimuli
                paradigms = self.paradigm_train

                params = []
                for p_nr, paradigm in enumerate(paradigms):
                    params.append(paradigm.present(repetitions=self.info['calibration_reps']))
                    self.info_trials[self.block_nr]['trials'][self.trial_nr][
                        'params_stim_' + self.info['paradigms']['train'][p_nr]] = params[-1]
                    check_escape(self.window)

                # decode with the training decoder (usually click)
                choice = self.decoder_train.decode(params[-1])
                self.info_trials[self.block_nr]['trials'][self.trial_nr]['kw_choice'] = choice
                check_escape(self.window)

                # highlight chosen box
                self.set_box_color(box_nr=choice, color='blue', colorSpace='named')
                self.draw(draw_answer=False)

                # set and output answer
                answer = keywords_show[choice]
                wav_file_answer = self.info['result_dir'] + 'audio/' + self.info[
                    'filename'] + f"_block{self.block_nr}_trial{self.trial_nr}_answer.wav"
                self.output_answer(answer=answer, wav_file=wav_file_answer)
                self.info_trials[self.block_nr]['trials'][self.trial_nr]['full_answer'] = answer
                self.info_trials[self.block_nr]['trials'][self.trial_nr]['answer_audio_file'] = wav_file_answer
                self.trial_nr += 1
                send_trig_if_parport(trigger=self.info['trigger']['trial_end'], parport=self.parport)

                if choice == idx_correct[n_trial]:
                    repeat = False
                    trials_correct.append(True)
                else:
                    trials_correct.append(False)
                    question_incorrect = "Incorrect, try again: " + question
                    self.set_question(question_incorrect)
                    print(question_incorrect)

                self.decoder_eval.pull_data(replace=False)

        print("--- Fitting decoder ---")
        self.decoder_eval.fit_calibration(trials_correct=trials_correct,
                                          stim_pres_order=idx_correct,
                                          paradigm_params=params[0],
                                          save_path=self.info['result_dir'] + self.info['filename'] + "_template.pkl")

        # trigger
        send_trig_if_parport(trigger=self.info['trigger']['calibration_block_end'], parport=self.parport)

        print("--- Calibration end ---")

    def accuracy_eval_trial(self, question, keywords, idx_correct):
        send_trig_if_parport(trigger=self.info['trigger']['trial_start'], parport=self.parport)

        self.info_trials[self.block_nr]['trials'][self.trial_nr] = {}

        # set question
        print(question)
        self.set_question(question)
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['question'] = question

        # set keywords
        self.load_keywords(keywords + [' '] * 6)
        keywords_show = self.set_keywords(which='first')
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['keywords'] = keywords_show

        paradigms = self.paradigm_eval
        decoder = self.decoder_eval
        block_type = 'eval'

        # clear decoder data and lsl buffer
        decoder.clear_data()

        # present stimuli
        params = []
        for p_nr, paradigm in enumerate(paradigms):
            params.append(paradigm.present(save_frame_intervals_path=None))
            self.info_trials[self.block_nr]['trials'][self.trial_nr][
                'params_stim_' + self.info['paradigms'][block_type][p_nr]] = params[-1]
            check_escape(self.window)

        choice = decoder.decode(params[-1], pull_new_data=True)

        self.info_trials[self.block_nr]['trials'][self.trial_nr]['kw_choice'] = choice
        self.info_trials[self.block_nr]['trials'][self.trial_nr]['kw_correct'] = idx_correct
        check_escape(self.window)

        save_json(dictionary=self.info_trials,
                  file_name=self.info['result_dir'] + self.info['filename'] + "_trials.json")

        # highlight chosen box
        self.set_box_color(box_nr=choice, color='blue', colorSpace='named')
        self.draw(draw_answer=False)

        send_trig_if_parport(trigger=self.info['trigger']['trial_end'], parport=self.parport)

        if choice == idx_correct:
            return True
        else:
            return False

    def accuracy_eval_block(self):
        print("--- Starting accuracy evaluation round ---")
        # trigger
        send_trig_if_parport(trigger=self.info['trigger']['accucary_eval_block_start'], parport=self.parport)

        self.set_scenario(text="")

        questions = self.info['texts']['text_calibration']['questions']
        keywords = self.info['texts']['text_calibration']['keywords']
        idxs_correct = self.info['texts']['text_calibration']['idx_correct']

        # instructions
        self.show_instructions(text=self.info['texts']['instruction_begin_acceval'])

        self.info_trials[self.block_nr] = {'scenario': 'fixed questions',
                                           'block_type': 'acceval',
                                           'trials': {}}
        change_bg_color(self.window, 'black')

        correct = []
        self.trial_nr = 0
        for i in range(len(questions)):
            correct.append(self.accuracy_eval_trial(question=questions[i],
                                                    keywords=keywords[i],
                                                    idx_correct=idxs_correct[i]))
            print(correct[-1])
            self.trial_nr += 1

        self.info_trials[self.block_nr]["accuracy"] = np.mean(correct)
        print(f"Accuracy: {np.mean(correct)}")
        save_json(dictionary=self.info_trials,
                  file_name=self.info['result_dir'] + self.info['filename'] + "_trials.json")

        send_trig_if_parport(trigger=self.info['trigger']['accucary_eval_block_end'], parport=self.parport)


    def run(self):
        if not self.debug_options["skip_intro"]:
            self.show_instructions(text=self.info['texts']['instruction_start'])

        # # resting state

        if not self.debug_options["skip_resting_state"]:
            resting_state(window=self.window, eyes_open=True, duration=self.info['timing']['rs_eyes_open'],
                          trig_start=self.info['trigger']['rs_open_start'],
                          trig_end=self.info['trigger']['rs_open_end'],
                          parport=self.parport, letter_height=self._instructions_letter_height)
            self.decoder_eval.fit_resting(trig_rest_start = self.info['trigger']['rs_open_start'],
                                          trig_rest_end=self.info['trigger']['rs_open_end'])
            resting_state(window=self.window, eyes_open=False, duration=self.info['timing']['rs_eyes_closed'],
                          trig_start=self.info['trigger']['rs_closed_start'],
                          trig_end=self.info['trigger']['rs_closed_end'],
                          parport=self.parport, letter_height=self._instructions_letter_height)

        # calibration

        if self.calibration:
            self.calibration_block()
            self.block_nr += 1

        # training blocks
        send_trig_if_parport(trigger=self.info['trigger']['training_start'], parport=self.parport)
        if not self.debug_options["skip_intro"]:
            self.show_instructions(text=self.info['texts']['instruction_begin_training'])
            paginated_text_with_pics(window=self.window,
                                     texts=self.info['texts']['instruction_task_descriptions'],
                                     pictures=self.info['texts']['instruction_task_descriptions_imgs'],
                                     bg_color=(0., 0., 0.),
                                     text_color=(1., 1., 1.),
                                     letter_height=self._instructions_letter_height)
        change_bg_color(self.window, 'black')
        while self.block_nr < self.n_blocks_calibrate + self.n_blocks_train:
            self.block(scenario=self.scenarios_train[self.block_nr - self.n_blocks_calibrate],
                       training=True)
            self.block_nr += 1
        send_trig_if_parport(trigger=self.info['trigger']['training_end'], parport=self.parport)

        # eval blocks
        send_trig_if_parport(trigger=self.info['trigger']['evaluation_start'], parport=self.parport)
        self.show_instructions(text=self.info['texts']['instruction_begin_eval'])
        change_bg_color(self.window, 'black')
        while self.block_nr < self.n_blocks_calibrate + self.n_blocks_train + self.n_blocks:
            self.block(scenario=self.scenarios_eval[self.block_nr - self.n_blocks_train - self.n_blocks_calibrate],
                       training=False)
            self.block_nr += 1
        send_trig_if_parport(trigger=self.info['trigger']['evaluation_end'], parport=self.parport)

        # accuracy evaluation
        self.accuracy_eval_block()

        # end resting state
        resting_state(window=self.window, eyes_open=False, duration=self.info['timing']['rs_eyes_closed'],
                      trig_start=self.info['trigger']['rs_closed_start'],
                      trig_end=self.info['trigger']['rs_closed_end'],
                      parport=self.parport, letter_height=self._instructions_letter_height)


        # exit experiment
        send_trig_if_parport(trigger=self.info['trigger']['experiment_end'], parport=self.parport)
        self.window.close()
        core.quit()


if __name__ == "__main__":
    # present dialog to collect info
    info = {'participant': 'testsubject',
            'session': 1,
            'experimenter': '',
            'language': 'en',
            'participant_screen': 2,
            }
    dlg = gui.DlgFromDict(info, sortKeys=False)
    if not dlg.OK:
        core.quit()

    # get settings from config file
    info, paradigm_train, paradigm_eval, decoder_train, decoder_eval, parport, lsl_inlet, debug_options = config(info, experiment=True)

    # generate window and screen layouts
    window = visual.Window(size=(1920, 1080),
                           monitor='testMonitor',
                           fullscr=True,
                           units=info['window_units'],
                           color=(0., 0., 0.),
                           colorSpace='rgb',
                           screen=info['participant_screen'],
                           )

    experiment = Experiment(window=window,
                            info=info,
                            paradigm_train=paradigm_train,
                            paradigm_eval=paradigm_eval,
                            decoder_train=decoder_train,
                            decoder_eval=decoder_eval,
                            run_calibration=info['run_calibration'],
                            parport=parport,
                            flicker_box=info['flicker_box'],
                            debug_options=debug_options
                            )

    experiment.run()




