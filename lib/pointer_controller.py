import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0
from os import path

# Manages controls based on a pointer coordinate like a mouse pointer or an eyetracker
class PointerController:

    coords = [0,0]
    screenSize = [0,0]
    origin_coords = [0,0]
    
    def __init__(self):
        self.screenSize = pyautogui.size()
        if (USE_COORDINATE_FILE == True and path.exists(COORDINATE_FILEPATH) == False):
            with open(COORDINATE_FILEPATH, 'w') as coordfile:
                coordfile.write("0,0")
                coordfile.close()
        
    def tick( self ):
        # Update the current coordinates
        if (USE_COORDINATE_FILE == True):        
            with open(COORDINATE_FILEPATH, 'r') as coordfile:
                raw_coords = coordfile.readline()
                file_coords = raw_coords.split(",")
                if (len(file_coords) == 2):
                    x = max( 0, int(float(file_coords[0])))
                    y = max( 0, int(float(file_coords[1])))
                    coords = [x,y]
                    self.update_coords( coords )
                coordfile.close()
        else:
            x, y = pyautogui.position()
            coords = [x,y]
            self.update_coords( coords )

    # Updates the current coordinates of our pointer
    def update_coords( self, coords ):
        self.coords = coords
        
    def update_origin_coords( self ):
        self.set_origin_coords( self.coords )
        
    # Set the origin point from which to do relative movements
    def set_origin_coords( self, coords ):
        self.origin_coords = coords
        
    # Set the origin point from which to do relative movements all the way to the right side of the screen
    def set_origin_coords_center_right( self ):
        self.origin_coords = [self.screenSize[0] + 400, self.screenSize[1] / 2]
        
    # Set the origin point from which to do relative movements all the way to the left side of the screen        
    def set_origin_coords_center_left( self ):
        self.origin_coords = [0 - 400, self.screenSize[1] / 2]
        
    # Detects in which quadrant the pointer is currently in
    # Counting goes as followed -
    # 1, 2, 3
    # 4, 5, 6
    # 7, 8, 9
    def detect_quadrant( self, widthSegments, heightSegments ):
        if( self.coords[0] <= 0 ):
            widthPosition = 0
        elif (self.coords[0] > self.screenSize[0]):
            widthPosition = widthSegments - 1
        else:
            widthPosition = math.floor( self.coords[0] / ( self.screenSize[0] / widthSegments ) )
        
        if( self.coords[1] <= 0 ):
            heightPosition = 0
        elif (self.coords[1] > self.screenSize[1]):
            heightPosition = heightSegments - 1
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
        
    # Detect in which direction the pointer is relative to the origin coordinate
    def detect_origin_directions( self, radius, y_radius=False, x_inversed=False, y_inversed=False ):
        x_diff = self.coords[0] - self.origin_coords[0]
        y_diff = self.coords[1] - self.origin_coords[1]
        if (x_inversed == True):
            x_diff = x_diff * -1
        if (y_inversed == True):
            y_diff = y_diff * -1            
        
        directions = []
        if( abs( x_diff ) > radius ):
            directions.append( "left" if x_diff < 0 else "right" )

        if (y_radius == False):
            y_radius = radius

        if( abs( y_diff ) > y_radius ):
            directions.append( "up" if y_diff < 0 else "down" )
            
        return directions
        
    def detect_origin_difference( self, dimension='x'):
        if (dimension == 'x'):
            return self.coords[0] - self.origin_coords[0]
        else:
            return self.coords[1] - self.origin_coords[1]            
        
    # Detect the coarse distance without using powers or square roots for performance optimalisations
    def detect_origin_coarse_distance( self, dimensions='xy' ):
        if (dimensions == 'xy'):
            x_diff = self.coords[0] - self.origin_coords[0]
            y_diff = self.coords[1] - self.origin_coords[1]
            
            return max( abs(x_diff), abs(y_diff) )
        else:
            return abs(self.detect_origin_difference(dimensions))

    # Detect if our pointer is inside an area
    def detect_area( self, x, y, width, height ):
        return self.coords[0] >= x and self.coords[1] >= y and self.coords[0] < x + width and self.coords[1] < y + height
    
    