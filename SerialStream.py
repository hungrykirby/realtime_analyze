import serial
import re
import numpy as np

import SvmSetup
import DL
import NeighborAlgorithm

def serial_loop(mode, stream, ver = ""):
    learner = None
    if mode == "dl":
        learner = DL
    elif mode == "svm":
        learner = SvmSetup
    else:
        learner = NeighborAlgorithm
    machine = learner.setup(ver)
    with serial.Serial('COM3',9600,timeout=0.1) as ser:
        bwn_a = False
        want_predict_num_array = []
        xyz_array = []
        while True and stream == "s":
            s = ser.readline()
            de = s.decode('utf-8')
            m = re.match("\-*[\w]+", str(de))
            if(m != None):
                num = m.group()
                #print(want_predict_num_array)
                if num == "a":
                    bwn_a = True
                elif bwn_a:
                    want_predict_num_array.append(int(num))
                if len(want_predict_num_array) == 3:
                    xyz = want_predict_num_array
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
