import serial
import re
import numpy as np
import sys

import SvmSetup
import DL
import NeighborAlgorithm

import config
import Calibrate

def serial_loop(mode, stream, file_type, ver = ""):
    learner = None
    if mode == "dl":
        learner = DL
    elif mode == "svm":
        learner = SvmSetup
    else:
        learner = NeighborAlgorithm
    machine = learner.setup(file_type, ver)
    if stream == "s":
        #with serial.Serial('COM3',9600,timeout=0.1) as ser:
        with serial.Serial('COM4',19200,timeout=0.1) as ser:
            bwn_a = False
            want_predict_num_array = []
            xyz_array = []
            count_label = 0
            while True:
                s = ser.readline()
                m = None
                try:
                    de = s.decode('utf-8')
                    m = re.match("\-*[\w]+", str(de))
                except Exception as e:
                    pass
                if(m != None):
                    num = m.group()
                    #print(want_predict_num_array)
                    if num == "a":
                        bwn_a = True
                        count_label = 0
                        want_predict_num_array = []
                    elif bwn_a:
                        want_predict_num_array.append(int(num))
                        #print(len(want_predict_num_array))
                    if len(want_predict_num_array) == int(file_type):
                        count_label = 0
                        if int(file_type) == 4:
                            xyz = want_predict_num_array - config.calibration_numbers_4
                        elif int(file_type) == 3:
                            xyz = want_predict_num_array - config.calibration_numbers_3
                        else:
                             print("Out of range:input_array", file_type)
                             sys.exit()
                        config.is_calibration = Calibrate.start_calibration(config.is_calibration, xyz)
                        while len(xyz_array) > 1:
                            xyz_array.pop(0)
                        xyz_array.append(xyz)
                        want_predict_num_array = []
                        bwn_a = False
                        ser.flushInput()
                        #print(xyz_array)
                        learner.stream(machine, np.array(xyz_array).astype(np.int64))
                        #SvmSetup.svm_stream(svc, np.array(xyz_array).astype(np.int64))
                else:
                    pass
        ser.close()
    else:
        print("no serial mode")
