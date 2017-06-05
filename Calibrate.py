import numpy as np
import config
import sys

def start_calibration(is_calibration, input_array):
    if config.count_calibration < config.FRAMES_CALIBRATION:
        if is_calibration:
            config.count_calibration += 1
            config.array_calibration.append(input_array)
            print("Now calibrating")
            #print("count_calibration", self.count_calibration)
    else:
        if len(input_array) == 4:
            config.calibration_numbers_4 = np.mean(np.array(config.array_calibration), axis=0)
        elif len(input_array) == 3:
            config.calibration_numbers_4 = np.mean(np.array(config.array_calibration), axis=0)
        else:
            print("Out of range:input_array", len(input_array))
            sys.exit()
        is_calibration = False
        config.count_calibration = 0
        config.array_calibration = []
        print("Calibration Finished")
    return is_calibration
