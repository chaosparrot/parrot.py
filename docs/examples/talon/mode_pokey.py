from lib.modes.base_mode import *
from lib.modes.base_talon_mode import BaseTalonMode

class PokeyMode(BaseTalonMode):
    patterns = [
        {
            'name': 'cluck_noise',
            'sounds': ['click_alveolar'],
            'threshold': {
                'percentage': 95,
                'power': 6000
            },
            'continual_threshold': {
                'power': 3000
            },
            # Uncomment this to disable Talon for 5 seconds after the noise has been made
            #'talon': {
            #    'throttle': 5.0
            #}
        }
    ]
    
    def handle_sounds( self, dataDicts ):
        #if( self.detect('cluck_noise') ):
            # Example - "Talon sleep" based on noise
            # self.talon_sleep()
            
            # Example - "Talon wake" based on noise
            # self.talon_wake()
            
            # Example - "Repeat that" based on noise
            # self.repeat_that()

            # Example - "Undo" based on noise
            # self.undo_that()
            
            # Example - Left click based on noise
            # self.touch()            
            
            # Example - Right click based on noise
            # self.righty()
            
            # Example scroll down and up based on how loud the sound is
            # momentum = (dataDicts[-1]['thrill_thr']['power'] - 3000 ) * 0.00005
            #self.wheel_down(momentum)
            #self.wheel_up(momentum)
        #else:
        super().handle_sounds(dataDicts) 