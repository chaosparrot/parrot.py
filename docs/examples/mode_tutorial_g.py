from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = [
        {
            'name': 'loud',
            'sounds': ['noise'],
            'threshold': {
                'percentage': 90,
                'power': 100000
            },
            'throttle': {
                'loud': 1.0,
                'chime': 1.0                
            }
        },
        {
            'name': 'chime',
            'sounds': ['noise'],
            'threshold': {
                'percentage': 90,
                'frequency': 200,
                'power': 70000
            },
            'throttle': {
                'chime': 1.0,
                'loud': 1.0                
            }
        }
    ]
    
    #### A - On the left, speech to detect, on the right, a list of buttons to press
    speech_commands = {
        'Computer press the a button': ['a'],
        'Computer stop speech recognition': ['b','exit'],
    }

    def handle_sounds( self, dataDicts ):
        if( self.detect('loud') ):
            #### B - Detect a loud sound to turn the speech recognition on ( Will press CTRL+Windows key, which automatically toggles windows speech recognition )                    
            self.toggle_speech()


    #### C - Detect a chiming sound to easily toggle speech recognition off            
    def handle_speech( self, dataDicts ):
        if( self.detect( 'chime' ) ):
            self.toggle_speech()