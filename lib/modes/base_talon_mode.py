from datetime import datetime
from lib.modes.base_mode import *
from abc import ABCMeta
from lib.talon_pipe import TalonPipe
import time

class BaseTalonMode(BaseMode, metaclass=ABCMeta):
    talon_throttled = False
    talon_throttle_timestamp = 0

    def start(self): 
        self.pipe = TalonPipe()
        self.pipe.connect()
        super().start()
    
    def exit(self):
        self.pipe.close()
        super().exit()
        
    def handle_sounds( self, dataDicts ):
        for pattern in self.patterns:
            if( self.detect(pattern["name"]) ):
                print( "DETECTED" )
                # Do talon specific things based on this pattern
                if ("talon" in pattern):
                    if ("throttle" in pattern['talon']):
                        self.talon_throttle_timestamp = time.time() + pattern['talon']['throttle']
                    if ("send" in pattern['talon'] and pattern['talon']['send'] == False):
                        continue

                print(f"{datetime.utcnow().isoformat()}: Sending {pattern['name']} to talon.")            
                self.send_user_action(pattern["name"])
                break
        
        # Change between talon wake and talon sleep if talon is throttled
        after_sound_throttled = time.time() < self.talon_throttle_timestamp
        if (after_sound_throttled != self.talon_throttled):
            if (after_sound_throttled == False):
                print( "Re-enabling Talon Voice")
                
                self.talon_wake()
                self.talon_throttled = False
            elif (after_sound_throttled == True):
                print( "Temporarily disabling Talon Voice")
                self.talon_sleep()
                self.talon_throttled = True

    def send_user_action(self, user_action):
        if (user_action.find('(') == -1 and user_action.find(')') == -1):
            user_action = user_action + '()'
        self.write('actions.user.'+ user_action)

    # Nicety functions that map to talon actions
    def talon_sleep(self):
        self.write("actions.speech.disable()")

    def talon_wake(self):
        self.write("actions.speech.enable()")

    def repeat_that(self):
        self.write("actions.core.repeat_command(1)")
        
    def undo_that(self):
        self.write("actions.edit.undo()")

    def touch(self):
        self.write("actions.mouse_click(0)")

    def righty(self):
        self.write("actions.mouse_click(1)")

    def write(self, action):
        self.pipe.write(action)

    def wheel_up(self, momentum=1):
        self.scroll_wheel(momentum * -1)
        
    def wheel_down(self, momentum=1):
        self.scroll_wheel(momentum)

    def scroll_wheel(self, momentum=1):
        scroll = int(momentum)
        self.write("actions.mouse_scroll(" + str(scroll) + ")")