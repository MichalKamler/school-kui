# Recipe from Python Cookbook, 2.23 Reading an Unbuffered Character in a Cross-Platform Way
try:
    from msvcrt import getwch as getch
except ImportError:
    """We are not on Windows; try the Unix-like approach"""

    def getch():
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


SKIP = False


def wait_n_or_s():
    "press n - next, s - skip to end ... write into terminal"
    global SKIP
    if SKIP:
        return
    while True:
        key = getch()
        print(key)
        if key == "n":
            break
        if key == "s":
            SKIP = True
            break
