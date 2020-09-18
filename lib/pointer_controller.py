import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Manages controls based on a pointer coordinate like a mouse pointer or an eyetracker
class PointerController:

    coords: [0,0]
    screenSize: [0,0]
    
    def __init__(self):
        self.screenSize = pyautogui.size()

    # Updates the current coordinates of our pointer
    def update_coords( self, coords ):
        self.coords = coords
        
    # Detects in which quadrant the pointer is currently in
    # Counting goes as followed -
    # 1, 2, 3
    # 4, 5, 6
    # 7, 8, 9
    def detect_quadrant( self, widthSegments, heightSegments ):
        if( self.coords[0] <= 0 ):
            widthPosition = 0
        elif (self.coords[0] > self.screenSize[0]):
            widthPosition = widthSegments
        else:
            widthPosition = math.floor( self.coords[0] / ( self.screenSize[0] / widthSegments ) )
        
        if( self.coords[1] <= 0 ):
            heightPosition = 0
        elif (self.coords[1] > self.screenSize[1]):
            heightPosition = heightSegments
        else:
            heightPosition = math.floor( self.coords[1] /( self.screenSize[1] / heightSegments ) )

        quadrant = 1 + widthPosition + ( heightPosition * widthSegments )
        return quadrant

    # Detects on what edges the pointer is currently in
    def detect_screen_edge( self, threshold ):        
        edges = []
        if( self.coords[1] <= threshold ):
            edges.append( "up" )
        elif( self.coords[1] >= self.screenSize[1] - threshold ):
            edges.append( "down" )
        
        if( self.coords[0] <= threshold ):
            edges.append( "left" )
        elif( self.coords[0] >= self.screenSize[0] - threshold ):
            edges.append( "right" )
            
        return edges

    # Detect if our pointer is inside an area
    def detect_area( self, x, y, width, height ):
        return self.coords[0] >= x and self.coords[1] >= y and self.coords[0] < x + width and self.coords[1] < y + height
    
    