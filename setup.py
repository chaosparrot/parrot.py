from config.config import *
from lib.record_data import record_sound
from lib.learn_data import learn_data
from lib.test_data import test_data

def root_navigation( first):
	if( first ):
		print( "Welcome to Parrot.PY setup!" )
		print( "----------------------------" )
	print( "Enter one of the buttons below and press enter to start" )
	print( " - [R] for recording" )
	print( " - [L] for learning the data" )
	print( " - [A] for analyzing the performance of the models" )
	print( " - [C] for general configuration" )
	print( " - [X] for exiting setup" )
	
	select_mode()

def select_mode():
	setup_mode = input("")
	if( setup_mode.lower() == 'r' ):
		record_sound()
		root_navigation( False )
	elif( setup_mode.lower() == 'l' ):
		learn_data()
		root_navigation( False )
	elif( setup_mode.lower() == 'a' ):
		test_data( True )
		root_navigation( False )		
	elif( setup_mode.lower() == 'c' ):
		print( "TODO CONFIG!" )
		root_navigation( False )
	elif( setup_mode.lower() == 'x' ):
		print( "Goodbye." )
	else:
		select_mode()
	
root_navigation( True )