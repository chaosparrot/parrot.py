from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
import time
from subprocess import call
import os
from lib.system_toggles import mute_sound, toggle_speechrec, toggle_eyetracker, turn_on_sound
import pandas as pd
import matplotlib.pyplot as plt

class TestMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher

	def start( self ):
		self.mode = "regular"
		self.centerXPos, self.centerYPos = pyautogui.position()
		toggle_eyetracker()
		mute_sound()
		self.testdata = []
		self.starttime = time.time()
		
		self.preventDoubleClickInPlotMode = time.time() 
		self.plot_in_seconds( 15.00 )

	def handle_input( self, dataDicts ):
	
		## Alter the data dicts into the right format for plotting
		dataRow = {'time': int((time.time() - self.starttime ) * 1000) / 1000 }
		for column in dataDicts[-1]:
			dataRow['intensity'] = dataDicts[-1][ column ]['intensity']
			dataRow[column] = dataDicts[-1][ column ]['percent']
			if( dataDicts[-1][ column ]['winner'] ):
				dataRow['winner'] = column
				
		if( self.mode == "regular" ):
			self.testdata.append( dataRow )
			
		## Allow any loud sound to click to let us close the plot once every second
		elif( dataRow['intensity'] > 2000 and ( time.time() - self.preventDoubleClickInPlotMode ) > 1 ):
			click()
			self.preventDoubleClickInPlotMode = time.time()
		
	def plot_in_seconds( self, time ):
		t = threading.Timer( time , self.display_results)
		t.daemon = True
		t.start()
		
	def display_results( self ):
		print( "Plotting results - Use any loud sound to click" )
		time.sleep( 2 )
		self.mode = "plotting"
		self.preventDoubleClickInPlotMode = time.time()
		
		plt.style.use('seaborn-darkgrid')
		palette = plt.get_cmap('Set1')
		num = 0
		bottom=0
		
		self.testdata = pd.DataFrame(data=self.testdata)

		# Add percentage plot
		plt.subplot(2, 1, 1)
		plt.title("Percentage distribution of predicted sounds", loc='left', fontsize=12, fontweight=0, color='black')
		plt.ylabel("Percentage")

		for column in self.testdata.drop(['winner', 'intensity', 'time'], axis=1):
			color = palette(num)
			if(column == "silence"):
				color = "w"
			if(column == "whistle"):
				color = "r"
			if(column == "hotel_bell"):
				color = "g"
			
			num+=1
			plt.bar(np.arange(self.testdata['time'].size), self.testdata[column], color=color, linewidth=1, alpha=0.9, label=column, bottom=bottom)
			bottom += np.array( self.testdata[column] )
		 
		plt.legend(loc=1, bbox_to_anchor=(1, 1.3), ncol=4)
		plt.subplot(2, 1, 2)
		
		# Add audio subplot
		plt.title('Audio', loc='left', fontsize=12, fontweight=0, color='black')
		plt.ylabel('Loudness')
		plt.xlabel("Time( s )")
		plt.ylim(ymax=40000)
		plt.plot(np.array( self.testdata['time'] ), np.array( self.testdata['intensity'] ), '-')
		plt.show()
		
		self.modeSwitcher.turnOnModeSwitch()				

	def exit( self ):
		toggle_eyetracker()
		turn_on_sound()
