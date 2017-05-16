import threading
import sys

import SerialStream

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


def loop():
    SerialStream.serial_loop(MODE, stream, ver)

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
    else:
        loop()


if __name__ == "__main__":
    main()
