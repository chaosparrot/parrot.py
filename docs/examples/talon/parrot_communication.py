from multiprocessing import shared_memory
from time import sleep
from talon import Module

mod = Module()
try:
    _parrotShm = shared_memory.SharedMemory(create=True, name="parrotpy_ipc", size=4096)
except FileExistsError:
    _parrotShm = shared_memory.SharedMemory(create=False, name="parrotpy_ipc", size=4096)
_parrotBuffer = _parrotShm.buf

IPC_STATE_PARROTPY_NOT_RUNNING = 0
IPC_STATE_PARROTPY_RUNNING = 1
IPC_STATE_PARROTPY_PAUSED = 2
IPC_STATE_PARROTPY_SWITCHING = 3
IPC_STATE_PARROTPY_SWITCH_AND_RUN = 4
IPC_STATE_PARROTPY_STOPPED = 5
IPC_STATE_PARROTPY_DISCONNECTED = 6

IPC_MEMLOC_PARROTPY_NEW_STATE = 1 # 8bit integer displaying the state we want parrot to transition to
IPC_MEMLOC_CLASSIFIER_LENGTH = 2 # 8bit integer of the length of the classifier string
IPC_MEMLOC_CLASSIFIER = 3 # 3 to 257 - UTF8 String contents ( max unicode length of 255 )
IPC_MEMLOC_CURRENT_MODE_LENGTH = 258 # 8bit integer of the length of the current running mode
IPC_MEMLOC_CURRENT_MODE = 259 # 259 to 513 - UTF8 String contents ( max unicode length of 255 )

@mod.action_class
class Actions:
    def parrot_resume():
        """Resumes parrot without blocking"""    
        _parrotBuffer[IPC_MEMLOC_PARROTPY_NEW_STATE] = IPC_STATE_PARROTPY_RUNNING
        
    def parrot_pause():
        """Pauses parrot"""    
        _parrotBuffer[IPC_MEMLOC_PARROTPY_NEW_STATE] = IPC_STATE_PARROTPY_PAUSED
        
    def parrot_stop():
        """Stops parrot noise recognition"""    
        print("Parrot flew into a window, it's super effective!")
        print("Parrot fainted")
        _parrotBuffer[IPC_MEMLOC_PARROTPY_NEW_STATE] = IPC_STATE_PARROTPY_STOPPED
        
    def parrot_request_switch():
        """Requests that parrot switches to the set classifier and mode"""    
        _parrotBuffer[IPC_MEMLOC_PARROTPY_NEW_STATE] = IPC_STATE_PARROTPY_SWITCHING
       
    def parrot_set_mode(mode_filename: str):
        """Sets the parrot mode and requests a switch to it"""
        mode_filename_in_bytes = mode_filename.encode('utf-8')
        strlen = len(mode_filename_in_bytes)
        if (strlen > 255):
            print( "Modes can have a maximum of 255 character length filenames")
            return

        # Save the overlay length and the image name at the same time to prevent race conditions
        _parrotBuffer[IPC_MEMLOC_CURRENT_MODE_LENGTH:IPC_MEMLOC_CURRENT_MODE + strlen] = bytes([strlen]) + mode_filename_in_bytes

    def parrot_set_classifier(classifier_name: str):
        """Sets the parrot noise classifier and requests a switch to it"""
        classifier_name_in_bytes = classifier_name.encode('utf-8')
        strlen = len(classifier_name_in_bytes)
        if (strlen > IPC_MAX_MEMLENGTH_STRING):
            print( "Classifiers can have a maximum of 255 character length filenames")
            return

        # Save the overlay length and the image name at the same time to prevent race conditions
        _parrotBuffer[IPC_MEMLOC_CLASSIFIER_LENGTH:IPC_MEMLOC_CLASSIFIER + strlen] = bytes([strlen]) + classifier_name_in_bytes
