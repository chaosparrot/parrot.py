from datetime import datetime
from lib.modes.base_mode import *
from abc import ABCMeta
from lib.talon_pipe import TalonPipe

class BaseTalonMode(BaseMode, metaclass=ABCMeta):
            
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
                if ("sendToTalon" in pattern and pattern["sendToTalon"] == False):
                    continue
                print(f"{datetime.utcnow().isoformat()}: Sending {pattern['name']} to talon.")            
                self.send_user_action(pattern["name"])

    def send_user_action(self, user_action):
        if (user_action.find('(') == -1 and user_action.find(')') == -1):
            user_action = user_action + '()'
        self.pipe.write('actions.user.'+ user_action)

    # Nicety functions that map to talon actions
    def talon_sleep(self):
        self.pipe.write("actions.speech.disable()")
        
    def talon_wake(self):
        self.pipe.write("actions.speech.enable()")

    def repeat_that(self):
        self.pipe.write("actions.core.repeat_command(1)")
        
    def undo_that(self):
        self.pipe.write("actions.edit.undo()")

    def touch(self):
        self.pipe.write("actions.mouse_click(0)")

    def righty(self):
        self.pipe.write("actions.mouse_click(1)")

    def wheel_up(self, momentum=1):
        self.scroll_wheel(momentum * -1)
        
    def wheel_down(self, momentum=1):
        self.scroll_wheel(momentum)

    def scroll_wheel(self, momentum=1):
        scroll = int(20 * momentum)
        self.pipe.write("actions.mouse_scroll(" + str(scroll) + ")")