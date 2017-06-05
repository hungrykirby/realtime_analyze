import threading
import sys

import SerialStream
import config

'''
console_input = sys.argv
if len(console_input) == 3:
    MODE = console_input[1].lower()
    stream = console_input[2].lower()
    ver = ""
elif len(console_input) == 4:
    MODE = console_input[1].lower()
    stream = console_input[2].lower()
    ver = console_input[3]
else:
    sys.exit()
'''
MODE = input("mode > ")
stream = input("is stream > ")
file_type = input("3 input sensors or 4 > ")
ver = input("version > ")

def loop():
    SerialStream.serial_loop(MODE, stream, file_type, ver)

if stream == "s":
    th = threading.Thread(target=loop,name="th",args=())
    th.setDaemon(True)
    th.start()

def main():
    if stream == "s":
        while True:
            tmp = input()
            if tmp == 'a':
                sys.exit()
            elif tmp == 'o':
                config.is_calibration = True
            else:
                print("no key")
    else:
        loop()


if __name__ == "__main__":
    main()
