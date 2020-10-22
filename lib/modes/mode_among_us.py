from config.config import *
from lib.modes.visual_mode import *
from lib.overlay_manipulation import update_overlay_image
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

class AmongUsMode(VisualMode):

    patterns = [
        {
            'name': 'kill',
            'sounds': ['sibilant_sh'],
            'threshold': {
                'percentage': 90,
                'power': 18000
            },            
            'throttle': {
                'kill': 0.5
            }
        },
        {
            'name': 'report',
            'sounds': ['sound_whistle'],
            'threshold': {
                'percentage': 95,
                'power': 100000
            },            
            'throttle': {
                'report': 0.5
            }
        },
        {
            'name': 'use',
            'sounds': ['click_alveolar'],
            'threshold': {
                'percentage': 82,
                'power': 12000
            },            
            'throttle': {
                'use': 0.5
            }
        },
        {
            'name': 'cancel',
            'sounds': ['fricative_f'],
            'threshold': {
                'percentage': 90,
                'power': 20000
            },
            'throttle': {
                'cancel': 0.5
            }
        },        
        {
            'name': 'drag',
            'sounds': ['sibilant_s'],
            'threshold': {
                'percentage': 90,
                'power': 4000
            },
            'continual_threshold': {
                'percentage': 20,
                'power': 2000
            }
        },
        {
            'name': 'mute',
            'sounds': ['sound_whistle'],
            'threshold': {
                'percentage': 95,
                'power': 30000
            },            
            'throttle': {
                'mute': 0.5
            }
        }
    ]
    
    def start(self):
        super().start()
        self.enable('single-press-grid')
        update_overlay_image('coords-overlay')
        self.toggle('play')
    
    hold_arrow_keys = []
        
    def handle_sounds( self, dataDicts ):
        if (self.detect('mute')):
            self.toggle('play')
            if ( len(self.hold_arrow_keys) > 0 ):
                for key in self.hold_arrow_keys:
                    self.release( key )
                self.hold_arrow_keys = []

    
        if (self.detect('play')):
            if (self.detect('drag')):
                self.drag_mouse()
            elif (self.detect('kill')):
                self.press('q')
            elif (self.detect('use')):
                self.press('space')
            elif (self.detect('cancel')):
                self.press('esc')
            
            self.handle_arrowkeys(dataDicts)

    def handle_arrowkeys( self, dataDicts, mode = "edges"):
        edges = self.handle_grid(mode)
        if ( len(self.hold_arrow_keys) > 0 ):
            for key in self.hold_arrow_keys:
                if ( key not in edges ):
                    self.release( key )

        if ( len(edges) > 0 ):
            for key in edges:
                if ( key not in self.hold_arrow_keys ):
                    self.hold( key )
        self.hold_arrow_keys = edges


    def handle_grid( self, mode = "edges" ):
        hold_arrowkeys = []
    
        if ( self.quadrant3x3 == 1 or self.quadrant3x3 == 4 or self.quadrant3x3 == 7 ):
            hold_arrowkeys.append( 'left' )
        elif (self.quadrant3x3 == 3 or self.quadrant3x3 == 6 or self.quadrant3x3 == 9 ):
            hold_arrowkeys.append( 'right' )
            
        if ( self.quadrant3x3 < 4 ):
            hold_arrowkeys.append( 'up' )
        elif ( self.quadrant3x3 > 6 ):
            hold_arrowkeys.append( 'down' )
            
        return hold_arrowkeys
