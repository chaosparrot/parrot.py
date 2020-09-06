from dragonfly import Grammar, CompoundRule
import pythoncom
from time import sleep
import pyautogui
from lib.modes import *
from lib.modes.mode_switch import SwitchMode
from lib.system_toggles import toggle_speechrec
import sys
import inspect
import importlib

class ModeSwitcher(object):
    __instance = None
    __currentMode = None    
    __modes = {}
    __is_testing = False
    
    def __new__(cls, is_testing=False):
        if ModeSwitcher.__instance is None:
            ModeSwitcher.__instance = object.__new__(cls)
            ModeSwitcher.__is_testing = is_testing
            
            ModeSwitcher.__modes = {
                'browse': BrowseMode(ModeSwitcher.__instance),
                'youtube': YoutubeMode(ModeSwitcher.__instance),
                'twitch': TwitchMode(ModeSwitcher.__instance),
                'switch': SwitchMode(ModeSwitcher.__instance),
                'heroes': HeroesMode(ModeSwitcher.__instance, is_testing),
                'starcraft': StarcraftMode(ModeSwitcher.__instance, is_testing),
                'phonemes': PhonemesMode(ModeSwitcher.__instance),                
                'testing': TestMode(ModeSwitcher.__instance),
                'worklog': ExcelMode(ModeSwitcher.__instance, ''),
                'excel': ExcelMode(ModeSwitcher.__instance, ''),
            }
            
        return ModeSwitcher.__instance
        
    def getMode(self):
        return ModeSwitcher.__currentMode
        
    def turnOnModeSwitch(self):
        self.switchMode( 'switch' )
                
    def switchMode( self, nextMode ):
        print( "Switching to " + nextMode )
        if( ModeSwitcher.__currentMode is not None ):
            ModeSwitcher.__currentMode.exit()
            
        if( nextMode not in self.__modes ):
            full_module_name = "lib.modes." + nextMode
            nextModule = importlib.import_module(full_module_name)
            
            module_found = False
            clsmembers = inspect.getmembers(sys.modules[full_module_name], inspect.isclass)
            for classname, actualclass in clsmembers:                
                if( actualclass.__module__ == full_module_name ):
                    module_found = True
                    self.__modes[nextMode] = actualclass(ModeSwitcher.__instance, self.__is_testing)
                    
            if( module_found ):
                ModeSwitcher.__currentMode = self.__modes[nextMode]
                ModeSwitcher.__currentMode.start()
            else:
                print( "MODE " + nextMode + " NOT FOUND!" )
        else:
            ModeSwitcher.__currentMode = self.__modes[nextMode]
            ModeSwitcher.__currentMode.start()
            
    def exit(self):
        if( ModeSwitcher.__currentMode is not None ):
            ModeSwitcher.__currentMode.exit()
