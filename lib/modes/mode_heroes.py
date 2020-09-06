from config.config import *
from lib.modes.visual_mode import *

class Heroes2Mode(VisualMode):

    previous_edges = []

    patterns = [
        {
            'name': 'ability 1',
            'sounds': ['vowel_ow', 'vowel_u'],
            'threshold': {
                'percentage': 90,
                'power': 30000
            },
            'throttle': {
                'ability 1': 0.2
            }
        },
        {
            'name': 'ability 2',
            'sounds': ['vowel_ae'],
            'threshold': {
                'percentage': 90,
                'power': 25000
            },
            'throttle': {
                'ability 2': 0.3
            }
        },
        {
            'name': 'ability 3',
            'sounds': ['fricative_f'],
            'threshold': {
                'percentage': 90,
                'power': 20000
            },            
            'throttle': {
                'ability 4': 0.3
            }
        },        
        {
            'name': 'ability 4',
            'sounds': ['general_vowel_aa', 'vowel_ah'],
            'threshold': {
                'percentage': 90,
                'power': 15000
            },            
            'throttle': {
                'ability 4': 0.3
            }
        },
        {
            'name': 'heroic',
            'sounds': ['approximant_r'],
            'threshold': {
                'percentage': 95,
                'power': 30000
            },
            'throttle': {
                'heroic': 0.5
            }
        },
        {
            'name': 'attack',
            'sounds': ['sound_whistle'],
            'threshold': {
                'percentage': 80,
                'power': 23000
            },            
            'throttle': {
                'attack': 0.5
            }
        },
        {
            'name': 'mount',
            'sounds': ['sibilant_sh'],
            'threshold': {
                'percentage': 90,
                'power': 18000
            },
            'throttle': {
                'mount': 0.5,
            }
        },        
        {
            'name': 'teleport',
            'sounds': ['vowel_e'],
            'threshold': {
                'percentage': 80,
                'power': 20000
            },
            'throttle': {
                'teleport': 0.5,
                'ability 2': 0.5
            }
        },
        {
            'name': 'control',
            'sounds': ['sibilant_z', 'fricative_v'],
            'threshold': {
                'percentage': 90,
                'power': 20000
            },             
            'throttle': {
                'control': 0.5
            }
        },
        {
            'name': 'numbers',
            'sounds': ['vowel_iy'],
            'threshold': {
                'percentage': 80,
                'power': 25000
            },            
            'throttle': {
                'numbers': 0.5
            }
        },
        {
            'name': 'camera',
            'sounds': ['vowel_y', 'vowel_ih'],
            'threshold': {
                'percentage': 60,
                'power': 20000
            },            
            'throttle': {
                'camera': 0.5
            }
        },
        {
            'name': 'move',
            'sounds': ['click_alveolar'],
            'threshold': {
                'percentage': 90,
                'power': 20000
            },
            'throttle': {
                'move': 0.1
            }
        },
        {
            'name': 'drag',
            'sounds': ['sibilant_s'],
            'threshold': {
                'percentage': 95,
                'power': 8000
            }
        },
        {
            'name': 'menu',
            'sounds': ['sound_call_bell'],
            'threshold': {
                'percentage': 80,
                'power': 100000
            },
            'throttle': {
                'menu': 1.0
            }
        },
        {
            'name': 'score',
            'sounds': ['sound_finger_snap'],
            'threshold': {
                'percentage': 90,
                'power': 38000
            },
            'throttle': {
                'score': 0.5
            }
        }        
    ]
    
    def arrowkey_camera( self, dataDicts ):
        edges = self.detector.detect_mouse_screen_edge( 40 )
        if ( len(edges) > 0 ):
            for key in edges:
                if ( key not in self.previous_edges ):
                    self.hold( key )
                    
        if ( len(self.previous_edges) > 0 ):
            for key in self.previous_edges:
                if ( key not in edges ):
                    self.release( key )
        self.previous_edges = edges
    
    def handle_sounds( self, dataDicts ):
        if( self.detect('drag') ):
            self.drag_mouse()
        else:
            self.stop_drag_mouse()

        self.arrowkey_camera( dataDicts )

        if( self.detect('move') ):
            self.rightclick()
        elif( self.detect('ability 1') ):
            self.press('q')
        elif( self.detect('ability 2') ):
            self.press('w')
        elif( self.detect('ability 3') ):
            self.press('e')
        elif( self.detect('ability 4') ):
            self.press('d')
        elif( self.detect('heroic') ):
            self.press('r')
        elif( self.detect('attack') ):
            self.press('a')            
        elif( self.detect('mount') ):
            self.press('z')
        elif( self.detect('teleport') ):
            self.press('b')            
        elif( self.detect('control') ):
            self.hold('ctrl')
        elif( self.detect('numbers') ):
            if (self.quadrant3x3 <= 3 ):
                self.press('1')
            elif (self.quadrant3x3 > 3 and self.quadrant3x3 <= 6 ):
                self.press('2')
            elif (self.quadrant3x3 == 7 or self.quadrant3x3 == 8 ):
                self.press('3')
            elif (self.quadrant3x3 == 9 ):
                self.press('4')

            self.release_special_keys()
        elif( self.detect('camera') ):
            if (self.quadrant3x3 == 5 ):
                self.press('space')
        elif( self.detect('menu') ):
            self.press('f10')
        elif( self.detect('score') ):
            self.press('tab')