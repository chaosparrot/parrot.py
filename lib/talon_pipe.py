import time
import os
import json
import socket
import sys

# This pipe is used to communicate with Talon Voice from parrot
# Most of this code is subject to change, as it relies on certain IPC possibilities of Talon
# As of Talon v0.1.5, the best way to communicate is using the named pipes defined in Talon
# In later versions of talon, a different IPC may be built
is_windows = sys.platform == 'win32'
if (is_windows):
    import win32file
    import win32pipe

    talon_pipe_location = r'\\.\pipe\talon_repl'
else:
    talon_pipe_location = os.path.expanduser('~/.talon/.sys/repl.sock')

class TalonPipe:

    # Checks for connection
    # If no connection is available, drop all writes until the connection is reestablished
    is_connected = False
    connection_attempt_timestamp = 0    
    reconnection_polling_threshold = 0.5

    win_pipe: None
    posix_pipe: None
    posix_writer: None

    def connect(self):
        self.connection_attempt_timestamp = time.time()    
        try:
            if( is_windows ):
                self.win_pipe = win32file.CreateFile( talon_pipe_location, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0,
                    None, win32file.OPEN_EXISTING, 0, None)
                    
                self.is_connected = True
            else:
                self.posix_writer = talon_pipe_location.makefile('w', buffering=1, encoding='utf8')            
                self.posix_pipe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.posix_pipe.connect( talon_pipe_location )
                self.is_connected = True

        except Exception as e:
            print( e )
            self.is_connected = False

    
    # Check if we have encountered an error somewhere connecting
    # If so, attempt a reconnect if the reconnection polling threshold is met
    def check_connection(self):
        if (self.is_connected == False and time.time() - self.connection_attempt_timestamp > self.reconnection_polling_threshold):
            print( "Attempting reconnect with Talon..." )
            self.connect()
            if (self.is_connected == True):
                print( "Talon connection established!" )

    # Write a command to the Talon Repl
    def write(self, data):
        self.check_connection()
    
        try:
            if (self.is_connected == True):
                data = json.dumps({'text': str(data) + '\n', 'cmd': 'input'})
                
                if (self.win_pipe is not None):
                    win32file.WriteFile(self.win_pipe, data.encode('utf8'))
                elif(self.posix_writer is not None):
                    self.posix_writer.write(data)
                    self.posix_writer.flush()
                    
        # When an error occurs, assume that the pipe has shut down for whatever reason
        except Exception as e:
            print( e )
            self.close()
    # Close the connection
    def close(self):
        self.is_connected = False
        try:
            if (self.win_pipe is not None):
                self.win_pipe.close()
                self.win_pipe = None
            elif (self.posix_pipe is not None):            
                 self.posix_pipe.shutdown(socket.SHUT_RDWR)
        except Exception: pass
