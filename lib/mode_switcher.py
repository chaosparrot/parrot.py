from dragonfly import Grammar, CompoundRule
from time import sleep
import pyautogui
from lib.modes.mode_switch import SwitchMode
from lib.system_toggles import toggle_speechrec
import sys
import inspect
import importlib
import lib.ipc_manager as ipc_manager
import os.path as path

class ModeSwitcher(object):
    __instance = None
    __currentMode = None
    __currentModeName = ""
    __modes = {}
    __is_testing = False
    
    def __new__(cls, is_testing=False):
        if ModeSwitcher.__instance is None:
            ModeSwitcher.__instance = object.__new__(cls)
            ModeSwitcher.__is_testing = is_testing
            ModeSwitcher.__modes = {}
            
        return ModeSwitcher.__instance
        
    def getMode(self):
        return ModeSwitcher.__currentMode
        
    def turnOnModeSwitch(self):
        self.switchMode( 'switch' )
                
    def switchMode( self, nextMode, run_after_switching = False ):
        # When no switch is needed - NOOP
        nextMode = nextMode.strip()
        if (self.__currentModeName == nextMode ):
            return True
        
        current_state = ipc_manager.getParrotState()
        if (run_after_switching == True):
            current_state = "running"
        
        ipc_manager.setParrotState("switching")
        print( "Switching to " + nextMode )
        if( ModeSwitcher.__currentMode is not None ):
            ModeSwitcher.__currentMode.exit()
            
        if( nextMode not in self.__modes ):
            # Keep the backwards compatible mode lib/modes/ directory for current users
            if (path.exists("lib/modes/" + nextMode + ".p")):
                full_module_name = "lib.modes." + nextMode 
            # Use data/code for new users
            elif (not path.exists("data/code/" + nextMode + ".py")):
                print("")            
                print("---- MODE NOT FOUND ERROR ----")
                print( "Could not find " + nextMode + ", does the " + nextMode + ".py file exist in the data/code folder?" )
                print("------------------------------")                
                exit()
            else:
                full_module_name = "data.code." + nextMode
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
                ipc_manager.setMode(nextMode)
                ipc_manager.requestParrotState(current_state)
                ipc_manager.setParrotState(current_state)
            else:
                print("")
                print("---- MODE NOT FOUND ERROR ----")
                print( "The file " + nextMode + ".py does not contain a valid class." )
                print( "Make sure it looks like one of the examples in the docs/examples folder." )
                print( "For more information on how modes work, look in the docs/TUTORIAL.md file." )
                print("------------------------------")
                exit()
                return False
        else:
            ModeSwitcher.__currentMode = self.__modes[nextMode]
            ModeSwitcher.__currentMode.start()
            ipc_manager.setMode(nextMode)
            ipc_manager.requestParrotState(current_state)
            ipc_manager.setParrotState(current_state)
        self.__currentModeName = nextMode
        return True

    def exit(self):
        if( ModeSwitcher.__currentMode is not None ):
            ModeSwitcher.__currentMode.exit()
        ipc_manager.setParrotState("stopped")
            
