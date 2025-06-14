from time import sleep
from pynput import keyboard
from pynput.keyboard import Key


stop_flag_f2 = False

def on_press_f2(key):
    global stop_flag_f2
    if key == Key.f2:
        stop_flag_f2 = True

f2_listener = keyboard.Listener(on_press=on_press_f2)
f2_listener.start()

def waiting():
    print("Press f2 to visualize result")
    while True:
        sleep(2)
        if stop_flag_f2:
            break
    f2_listener.stop()