import time
import pickle
import numpy as np
from config import config
from shared_memory_dict import SharedMemoryDict
from util.eeg.online import lsl_trigger_decode
from util.helper import create_folder_if_not_exist

if __name__ == "__main__":

    load_model = None  # to recover from a given model instead of training, provide path here
    #load_model = "../data/experiments/subject1_2023-07-10_09-40-40/model_refitted.keras"

    # get settings from config file
    info, decoder_class_continuous, decoder_continuous_kwargs, lsl_inlet = config({}, experiment=False)

    # create shared dictionary
    shared_dict = SharedMemoryDict(name=info['shared_dict_name'], size=1024*2)
    shared_dict["decoder_started"] = True

    # initialize decoder
    try:
        decoder = decoder_class_continuous(lsl_inlet=lsl_inlet, **decoder_continuous_kwargs)
    except TypeError:
        raise AttributeError("The decoder_main.py script is only valid and necessary for continuous decoders and the "
                             "respective paradigms. Check the value of decoder_class_continuous in config.py.")

    # wait until the experiment was initialized, then get paradigm information from shared dictionary
    print("Waiting for experiment start trigger...")
    decoder.wait_for_trig(triggers=info['trigger']['experiment_start'],
                          wait_before_update_s=0.5)
    paradigm_params = shared_dict['decode_params']
    samples_per_trial = int(len(paradigm_params['stim_sequences']) / paradigm_params['frame_nr_multiplier'] * paradigm_params['max_repetitions'])
    result_dir = shared_dict["result_dir"]
    create_folder_if_not_exist(result_dir + "/data")

    print("Experiment started")

    if load_model is None:
        # wait for first calibration or training trigger, then calibrate
        print("Waiting for calibration start or training start trigger...")
        decoder.wait_for_trig(triggers=[info['trigger']['calibration_block_start'],
                                        info['trigger']['training_start']],
                              wait_before_update_s=0.5)

        print("Calibration started")
        verbosity = 1
        while not decoder.has_trig_happened(info['trigger']['evaluation_start']):
            # get eeg data
            print("...pulling training data")
            decoder.pull_training_data_X(paradigm_params, wait_before_update_s=0.5,
                                         save_path=result_dir + "/data/training_data_X.pkl")

            print("...waiting for click")
            # time.sleep(0.5)
            # wait until user has clicked, then set correct labels
            y = None
            while y is None:
                try:
                    clicked = shared_dict['clicked']
                    y = clicked
                except KeyError:
                    pass
                time.sleep(0.05)
            print(f"clicked: {y}")
            shared_dict['clicked'] = None

            decoder.set_training_data_y(y, paradigm_params, save_path=result_dir + "/data/training_data_y.pkl")

            # fit model
            decoder.fit_model(epochs=50, batch_size=30, from_scratch=False, n_val=samples_per_trial,
                                  save_path=result_dir + "model.keras", verbosity=verbosity)

        print("Calibration finished, refitting model on all data.")
        decoder.fit_model(epochs=50, batch_size=30, from_scratch=True, n_val=None,
                          save_path=result_dir + "model_refitted.keras", verbosity=verbosity)
        print("Done.")
    else:
        print("Loading model.")
        decoder.load_fitted_model(path=load_model)
        print("Done.")

    rep = 0
    while True:
        print("Decoding...")
        decoded_result = decoder.decode(paradigm_params=paradigm_params,
                                        wait_before_update_s=0.01,
                                        special_stim_id=9,  # Make extra sure the person wants to click finished
                                        special_stim_frame_wait=10,
                                        save_path=result_dir + f"/data/test_data_X_{rep}.pkl")
        shared_dict['decoded'] = decoded_result
        print(f"Result: {decoded_result}")
        rep += 1




