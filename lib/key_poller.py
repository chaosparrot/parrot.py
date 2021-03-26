# From https://stackoverflow.com/a/31736883
from config.config import IS_WINDOWS

if (IS_WINDOWS == True):
    import msvcrt
else:
    import sys
    import select
    import termios

# Courtesy from pokeyrule (https://github.com/pokey)
class KeyPoller():
    def __enter__(self):
        if (IS_WINDOWS == False):
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        return self

    def __exit__(self, type, value, traceback):
        if(IS_WINDOWS == False ):
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def poll(self):
        if( IS_WINDOWS == True ):
            if( msvcrt.kbhit() ):
                return msvcrt.getch().decode()
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
        return None