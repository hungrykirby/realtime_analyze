import threading
import sys

import SerialStream

console_input = sys.argv
if len(console_input) == 3:
    MODE = console_input[1].lower()
    stream = "s"
    ver = ""
elif len(console_input) == 3:
    MODE = console_input[1].lower()
    stream = "n"
    ver = console_input[2]
else:
    sys.exit()


def loop():
    SerialStream.serial_loop(MODE, stream, ver)

th = threading.Thread(target=loop,name="th",args=())
th.setDaemon(True)
th.start()

def main():
    while True:
        tmp = input()
        if tmp == 'a':
            sys.exit()


if __name__ == "__main__":
    main()
