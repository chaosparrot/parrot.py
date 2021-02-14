from multiprocessing import shared_memory
import array
import time
import struct

# ----------------- NOTICE ---------------- 
# THIS FILE IS USED TO ENABLE OTHER PROGRAMS TO INTERACT OR TO READ OUT STATES FROM PARROT.PY
# CHANGE THESE MEMORY ADDRESSES AS LITTLE AS POSSIBLE TO PREVENT ISSUES ARISING
# -----------------------------------------

IPC_MAX_MEMLENGTH_STRING = 255

# Shared memory locations for overlay and control visualisation looks like this
IPC_MEMLOC_CTRL_STATE = 0 # Ctrl pressed down = 1, else 0
IPC_MEMLOC_SHIFT_STATE = 1 # Shift pressed down = 1, else 0
IPC_MEMLOC_ALT_STATE = 2 # Alt pressed down = 1, else 0
IPC_MEMLOC_UP_STATE = 3 # Up pressed down = 1, else 0
IPC_MEMLOC_DOWN_STATE = 4 # Down pressed down = 1, else 0
IPC_MEMLOC_LEFT_STATE = 5 # Left pressed down = 1, else 0
IPC_MEMLOC_RIGHT_STATE = 6 # Right pressed down = 1, else 0
IPC_MEMLOC_OVERLAY_LENGTH = 7 # 8bit integer of the length of overlay filename
IPC_MEMLOC_OVERLAY_FILENAME = 8 # 8 to 262 - UTF8 String contents ( max unicode length of 255 )
IPC_MEMLOC_SOUNDNAME_LENGTH = 263 # 8bit integer of the length of the sound string
IPC_MEMLOC_SOUNDNAME = 264 # 264 to 519 - UTF8 String contents ( max unicode length of 255 )
IPC_MEMLOC_ACTIONNAME_LENGTH = 520 # 8bit integer of the length of the action string
IPC_MEMLOC_ACTIONNAME = 521 # 521 to 775 - UTF8 String contents ( max unicode length of 255 )
IPC_MEMLOC_ACTION_AMOUNT = 776 # 776 to 777 - 16bit integer of the amount of times this action has been repeated

_shm = None
_buffer = None
try:
    _shm = shared_memory.SharedMemory(create=True, name="parrotpy_ui", size=4096)
except FileExistsError:
    _shm = shared_memory.SharedMemory(create=False, name="parrotpy_ui", size=4096)
_buffer = _shm.buf

# A map of all the button states memory locations
_ipc_button_state = {
    'ctrl': IPC_MEMLOC_CTRL_STATE,
    'shift': IPC_MEMLOC_SHIFT_STATE,
    'alt': IPC_MEMLOC_ALT_STATE,
    'up': IPC_MEMLOC_UP_STATE,
    'down': IPC_MEMLOC_DOWN_STATE,
    'left': IPC_MEMLOC_LEFT_STATE,
    'right': IPC_MEMLOC_RIGHT_STATE
}

def setButtonState( button, state ):
    if (button in _ipc_button_state):
        _buffer[_ipc_button_state[button]] = 1 if state > 0 else 0
        
def getButtonState( button ):
    if (button in _ipc_button_state):
        return _buffer[_ipc_button_state[button]] > 0
    else:
        return False

def setOverlayImage( filename ):
    overlayimage_in_bytes = filename.encode('utf-8')
    strlen = len(overlayimage_in_bytes)
    if (strlen > IPC_MAX_MEMLENGTH_STRING):
        print( "Overlay images can have a maximum of 255 character length filenames")
        return

	# Save the overlay length and the image name at the same time to prevent race conditions
    _buffer[IPC_MEMLOC_OVERLAY_LENGTH:IPC_MEMLOC_OVERLAY_FILENAME + strlen] = bytes([strlen]) + overlayimage_in_bytes

def getOverlayImage():
    strlen = _buffer[IPC_MEMLOC_OVERLAY_LENGTH]
    if (strlen > 0):
        return array.array('B', _buffer[IPC_MEMLOC_OVERLAY_FILENAME:IPC_MEMLOC_OVERLAY_FILENAME + strlen]).tobytes().decode('utf-8')    
    else:
        return ""
        
def setSoundName( soundname ):
    soundname_in_bytes = soundname.encode('utf-8')
    strlen = len(soundname_in_bytes)
    if (strlen > IPC_MAX_MEMLENGTH_STRING):
        print( "Sound names can have a maximum of 255 character length")
        return        

	# Save the sound name length and the sound name at the same time to prevent race conditions
    _buffer[IPC_MEMLOC_SOUNDNAME_LENGTH:IPC_MEMLOC_SOUNDNAME + strlen] = bytes([strlen]) + soundname_in_bytes

def getSoundName():
    strlen = _buffer[IPC_MEMLOC_SOUNDNAME_LENGTH]
    if (strlen > 0):
        return array.array('B', _buffer[IPC_MEMLOC_SOUNDNAME:IPC_MEMLOC_SOUNDNAME + strlen]).tobytes().decode('utf-8')    
    else:
        return ""
        
def setActionName( actionname ):
    actionname_in_bytes = actionname.encode('utf-8')
    strlen = len(actionname_in_bytes)
    if (strlen > IPC_MAX_MEMLENGTH_STRING):
        print( "Action names can have a maximum of 255 character length")
        return
    
    total_action_bytes = bytes([strlen]) + actionname_in_bytes + bytes(IPC_MAX_MEMLENGTH_STRING - strlen)
    total_action_length = len(total_action_bytes)
    
    # If the action is the same, increment the amount
    if (_buffer[IPC_MEMLOC_ACTIONNAME_LENGTH:IPC_MEMLOC_ACTIONNAME_LENGTH + total_action_length] == total_action_bytes):
        integeramount = struct.unpack('>H', _buffer[IPC_MEMLOC_ACTION_AMOUNT:IPC_MEMLOC_ACTION_AMOUNT + 2].tobytes())[0] + 1
        intbytes = struct.pack('>H', integeramount)
    else:
        intbytes = struct.pack('>H', 1)
        
	# Save the action name length, the action name and the action amount at the same time to prevent race conditions
    _buffer[IPC_MEMLOC_ACTIONNAME_LENGTH:IPC_MEMLOC_ACTIONNAME_LENGTH + total_action_length + 2] = total_action_bytes + intbytes

def getActionName():
    strlen = _buffer[IPC_MEMLOC_ACTIONNAME_LENGTH]
    if (strlen > 0):
        return array.array('B', _buffer[IPC_MEMLOC_ACTIONNAME:IPC_MEMLOC_ACTIONNAME + strlen]).tobytes().decode('utf-8')    
    else:
        return ""
        
def getActionAmount():
    return struct.unpack('>H', _buffer[IPC_MEMLOC_ACTION_AMOUNT:IPC_MEMLOC_ACTION_AMOUNT + 2])[0]