#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on August 11, 2025, at 14:00
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""
# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
# importing modules
import os
import pandas as pd
import numpy as np
import tobii_research as tr
import random

os.chdir(os.path.dirname(__file__))
time_in_roi = 0

r = np.zeros(3)
null_trial = False
end_image_display = False
missed_trials = 0  # Ensure this is present
buffer = 30

found_eyetrackers = tr.find_all_eyetrackers()
Eyetracker = found_eyetrackers[0]
gaze_data_buffer = []
gaze_in_fixation = False

x_left = 0
x_right = 0
y_bottom = 0
y_top = 0

callback_mode = 'fixation'

trialCount = 0
trial_info = {}
current_trial_phase = 'setup'
fullExpData = None

def gaze_data_callback(gaze_data):
    """
    Callback function that changes behavior based on global state
    """
    global gaze_in_fixation, null_trial, end_image_display, gaze_data_buffer, _winSize, callback_mode
    global trialCount, current_trial_phase, trial_info
    
    gaze_in_fixation = null_trial = end_image_display = False
    
    try:
       left_eye = gaze_data.left_eye
       right_eye = gaze_data.right_eye
       left_validity = left_eye.gaze_point.validity == 1
       right_validity = right_eye.gaze_point.validity == 1
       
       t = gaze_data.system_time_stamp / 1000.0
       gx, gy, gv = (-1, -1, 0)
        
       if left_validity and right_validity:
            gv = 1
            left_pos = left_eye.gaze_point.position_on_display_area
            right_pos = right_eye.gaze_point.position_on_display_area
            gx = ((left_pos[0] + right_pos[0]) / 2) * _winSize[0]
            gy = _winSize[1] - (((left_pos[1] + right_pos[1]) / 2) * _winSize[1])
       elif left_validity:
            gv = 1
            pos = left_eye.gaze_point.position_on_display_area
            gx = pos[0] * _winSize[0]
            gy = _winSize[1] - (pos[1] * _winSize[1])
       elif right_validity:
            gv = 1
            pos = right_eye.gaze_point.position_on_display_area
            gx = pos[0] * _winSize[0]
            gy = _winSize[1] - (pos[1] * _winSize[1])
        
       trial_info = {
           'trial_number': [trialCount],
           'phase': [current_trial_phase],  # 'fixation', 'image_display', 'input', 'feedback'
           'callback_mode': [callback_mode]
       }
       gaze_data_buffer.append((t, gx, gy, gv, trialCount, current_trial_phase, callback_mode))

       # Behavior depends on current mode
       if callback_mode == "fixation":
           if gv == 1:
                if not (x_left <= gx <= x_right and y_bottom <= gy <= y_top):
                    gaze_in_fixation = False
                else:
                    gaze_in_fixation = True
           else:
                gaze_in_fixation = False
            
                
       elif callback_mode == "image_display":
           if gv == 1:
                if not (x_left <= gx <= x_right and y_bottom <= gy <= y_top):
                    gaze_in_fixation = False
                    null_trial = True
                    end_image_display = True
                    
                else:
                    gaze_in_fixation = True
                    end_image_display = False
                    
           else:
                gaze_in_fixation = False
                null_trial = True
                end_image_display = True
                
       results = (gaze_in_fixation, end_image_display, null_trial)
       return results
            
    except Exception as e:
        print(f"Error in gaze_data_callback: {e}")

def write_buffer_to_file(buffer, output_path):
    # Make a copy of the buffer and clear it
    buffer_copy = buffer[:]
    buffer.clear()
    # Define column names
    columns = ['trial_num', 'time', 'gaze_x', 'gaze_y', 'validity', 'phase', 'callback_mode']
    # Convert buffer to DataFrame
    out = pd.DataFrame(buffer_copy, columns=columns)
    # Check if the file exists
    file_exists = not os.path.isfile(output_path)
    # Write the DataFrame to a CSV file
    out.to_csv(output_path, mode='a', index =False, header = file_exists)

# ROI Function
def get_area_of_interest(screen_resolution, area_of_interest, position_of_interest):
    """
    Visualizes the area of interest within the given screen resolution.
    
    Parameters:
    screen_resolution (list): A list containing the width and height of the screen resolution [width, height].
    area_of_interest (list): A list containing the width and height of the area of interest [width, height].
    position_of_interest (list): A list containing the x and y coordinates of the position of the area of interest relative to the center [x, y].
    
    Returns:
    list: A list containing the start and end coordinates of the area of interest rectangle [x_start, x_end, y_start, y_end].
    """
    screen_width, screen_height = screen_resolution
    aoi_width, aoi_height = area_of_interest
    aoi_x, aoi_y = position_of_interest
    
    # Calculate the x and y coordinates of the area of interest
    x_start = screen_width // 2 + aoi_x - aoi_width // 2
    x_end = x_start + aoi_width
    y_start = screen_height // 2 + aoi_y - aoi_height // 2
    y_end = y_start + aoi_height
    
    # Ensure the area of interest stays within the screen resolution
    x_start = max(0, min(x_start, screen_width - aoi_width))
    x_end = max(0, min(x_end, screen_width))
    y_start = max(0, min(y_start, screen_height - aoi_height))
    y_end = max(0, min(y_end, screen_height))
    
    return [x_start, x_end, y_start, y_end]
# Run 'Before Experiment' code from Code_ImageDisplay
import random
import pandas as pd

# Experiment Class
# There is only 1 instance of this for the full experiment
class experimentData: # not the same as PsychoPy's "thisExp.data"!!!
    
    def __init__(self):
        
        # Experiment Global Variables (default values)
        self.readGlobalVars()
        
        # Experiment Conditions
        self.targetOrientList = list()
        self.distPresentList = list()
        self.distTypeList = list()
        self.imgLocationList = list() # to store where the image is located relative to the centre
        self.trialBlockList = list()
        self.pseudoRandomExpCond()
        
        '''
        print("DEBUG STATEMENTS")
        print(self.targetOrientList)
        print(self.distPresentList)
        print(self.distTypeList)
        '''
    
    # Reads global variables from "" Excel File
    def readGlobalVars(self):
        globalVarDF = pd.read_csv('GlobalVariables.csv', header = 0)
        # print(globalVarDF.to_string())
        self.totalTrials = globalVarDF["TotalTrials"].values[0]
        self.totalImageCount = globalVarDF["TotalImages"].values[0]
        self.fillerTime = globalVarDF["ImageDisplayTime"].values[0] / 1000 # input in ms, saved as s
        self.targetTime = globalVarDF["ImageDisplayTime"].values[0] / 1000
        self.distractorTime = globalVarDF["ImageDisplayTime"].values[0] / 1000
        self.targetOffsets = globalVarDF["TargetOffset"].values[0].split("_") #eg. 1_2_8 = ['1','2','8']
        self.targetOffsets = list(int(x) for x in self.targetOffsets)
        self.trialBlocks = globalVarDF['TrialBlocks'].values[0].split('_')
    
    # Generates a pseudo-randomised list for all possible conditions in a trial
    def pseudoRandomExpCond(self):
        
        # Generate pseudo-random list for all trials in experiment
        
        # Target Orientation
        targetOrient = ["LEFT","RIGHT"]
        self.targetOrientList = targetOrient * int(self.totalTrials/2)
        if(self.totalTrials % 2 != 0): # handle odd trial count length
            rng = random.randint(0,1)
            self.targetOrientList.append(targetOrient[rng])
        
        # Image Blocks (Centre or Periphery)
        self.trialBlockList = (self.trialBlocks * int(self.totalTrials / len(self.trialBlocks)))
        diff = self.totalTrials - len(self.trialBlockList)
        for i in range(diff):
            rng = random.randint(0, len(self.trialBlocks) - 1)
            self.trialBlockList.append(self.trialBlocks[rng])
        
        # Image Position on Screen Relative to Centre
        imagePos = ["LEFT", "RIGHT"]
        self.imgLocationList = (imagePos * int(len(self.trialBlockList) / len(imagePos)))
        diff = self.totalTrials - len(self.imgLocationList)
        for i in range(diff):
            rng = random.randint(0, len(imagePos) - 1)
            self.imgLocationList.append(imagePos[rng])
            
        # Distractor Present, Distractor Type
        self.distPresentList = [True] * self.totalTrials # always present here!
        distType = ["NEGATIVE","POSITIVE","NEUTRAL"]
        self.distTypeList = (distType * int(len(self.trialBlockList)/len(distType)))
        diff = self.totalTrials - len(self.distTypeList) # handle numbers not divisible by count of dist types
        for i in range(diff):
            rng = random.randint(0,len(distType) - 1)
            self.distTypeList.append(distType[rng])
            
        # Target Distractor Lag
        self.targetOffsetList = (self.targetOffsets * int(len(self.distTypeList)/len(self.targetOffsets)))
        diff = self.totalTrials - len(self.targetOffsetList) # handle numbers not divisible by count of targetOffsets
        for i in range(diff):
            rng = random.randint(0,len(self.targetOffsets) - 1)
            self.targetOffsetList.append(self.targetOffsets[rng])
            
        # shuffle them around in randomised order
        random.shuffle(self.targetOrientList)
        random.shuffle(self.distPresentList)
        random.shuffle(self.distTypeList)
        random.shuffle(self.targetOffsetList)
        random.shuffle(self.imgLocationList)
        random.shuffle(self.trialBlockList)
        
# Run 'Before Experiment' code from Code_InputPhase
accuracyText = "DEFAULT"

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'nikita_test_2'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1200]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\nikit\\OneDrive\\Desktop\\TEST_RUN\\nikita_test_2_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    #global ioServer
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('KeyInput_Instructions') is None:
        # initialise KeyInput_Instructions
        KeyInput_Instructions = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='KeyInput_Instructions',
        )
    if deviceManager.getDevice('KeyInput_InputPhase') is None:
        # initialise KeyInput_InputPhase
        KeyInput_InputPhase = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='KeyInput_InputPhase',
        )
    if deviceManager.getDevice('KeyInput_ExitScreen') is None:
        # initialise KeyInput_ExitScreen
        KeyInput_ExitScreen = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='KeyInput_ExitScreen',
        )
    # return True if completed successfully
    return True


def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    
    # IMPORTANT: Make sure to calibrate the eyetracker, or it won't provide accurate data
    calibration = tr.ScreenBasedCalibration(Eyetracker)
    calibration.enter_calibration_mode()
    print("Entered calibration mode")
    calibration_stim = visual.Circle(win, radius=0.02, units='norm', fillColor='red')
    points_to_calibrate = [(0.5, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
    for point in points_to_calibrate:
        calibration_stim.pos = [(point[0]-0.5)*2, (point[1]-0.5)*2]
        calibration_stim.draw()
        win.flip()
        
        print(f"Show a point on the screen at {point}")
        core.wait(0.7)
        
        print(f"Collecting data at {point}")
        if calibration.collect_data(point[0], point[1]) != tr.CALIBRATION_STATUS_SUCCESS:
            calibration.collect_data(point[0], point[1])
            
    print('Computing and applying calibration')
    calibration_result = calibration.compute_and_apply()
    print(f"Computing returned {calibration_result.status} collected at {len(calibration_result.calibration_points)} points")
    calibration.leave_calibration_mode()
    print('Left calibration mode')
    
    print(f"Calibration status: {calibration_result.status}")
    win.flip()
    
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "load_functions" ---
    # Run 'Begin Experiment' code from code
    global x_left, x_right, y_bottom, y_top
    x_left, x_right, y_bottom, y_top = get_area_of_interest(screen_resolution=[1920, 1200], area_of_interest=[100, 100], position_of_interest=[0,0])
    print(x_left, x_right, y_bottom, y_top)
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    Text_Instructions = visual.TextStim(win=win, name='Text_Instructions',
        text="Welcome to the experiment!\n\nYou will be shown a sequence of upright images. Out of all the images there will be one image with a different orientation, either tilted (90°) towards the left, or tilted(90°) towards the right. \n\nAt the end of the trial you have to indicate the orientation of the tilted image by pressing the 'LEFT' key for left orientation or\nthe 'RIGHT' key for right orientation.\n\nPress the 'SPACE' key to begin.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    KeyInput_Instructions = keyboard.Keyboard(deviceName='KeyInput_Instructions')
    
    
    # --- Initialize components for Routine "Fixation" ---
    Fixation_Shape = visual.ShapeStim(
        win=win, name='Fixation_Shape', vertices='cross',units='pix', 
        size=(25, 25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "ImageDisplay" ---
    # Run 'Begin Experiment' code from Code_ImageDisplay
    # Data File for Export
    global fullExpData
    fields = ["Trial No.", "Block", "Target Image","Target Position","Target Orientation","Distractor Position","Distractor Type","Distractor Image","Target-Distractor Lag","Accuracy","Response Time"]
    fullExpData = pd.DataFrame(list(), columns = fields)
        
    # ! Start Experiment !
    trialCount = 0
    thisExperimentData = experimentData() # not the same as PsychoPy's "thisExp.data"!!!
    
    # Experiment Constants
    totalTrials = thisExperimentData.totalTrials
    totalImageCount = thisExperimentData.totalImageCount
    fillerTime = thisExperimentData.fillerTime
    targetTime = thisExperimentData.targetTime 
    distractorTime = thisExperimentData.distractorTime
    targetOffsets = thisExperimentData.targetOffsets
    blockList = thisExperimentData.trialBlockList
    
    # Define image component
    ImageComponent = visual.ImageStim(
        win=win,
        name='ImageComponent', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(550, 450),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # Image Selection Utilities
    def pickTargetImg(orientation): # can be "LEFT" or "RIGHT"
        
        # ensure correct input
        validOrientation = ["LEFT", "RIGHT"]
        if(orientation not in validOrientation):
            return "INVALID ORIENTATION"
        
        # pick image at random based on orientation
        # NEED TO PREVENT REPETITIONS
        targetImg = "Images\\Target\\"
        if(orientation == "LEFT"):
            targetImg = targetImg + "L" + str(random.randint(1, 79)) + ".jpg"
        elif(orientation == "RIGHT"):
            targetImg = targetImg + "R" + str(random.randint(1, 80)) + ".jpg"
        
        return targetImg
        
    # Filler Image Helper Function
    def pickFillerImg():
        
        # picks image at random
        # NEED TO PREVENT REPETITIONS
        fillerImg = "Images\\Fillers\\" + str(random.randint(1, 123)) + ".jpg"
        return fillerImg
    
    # Distractor Image Helper Function
    def pickDistImg(distType): # can be "NEGATIVE" or "POSITIVE" or "NEUTRAL"
        
        # ensure correct input
        validType = ["NEGATIVE", "POSITIVE", "NEUTRAL"]
        if(distType not in validType):
            return "INVALID TYPE"
        
        # pick image at random based on orientation
        # NEED TO PREVENT REPETITIONS
        distImg = "Images\\"
        if(distType == "NEGATIVE"):
            distImg = distImg + "DistNeg\\" + str(random.randint(1, 90)) + ".jpg"
        elif(distType == "POSITIVE"):
            distImg = distImg + "DistPos\\" + str(random.randint(1, 90)) + ".jpg"
        elif(distType == "NEUTRAL"):
            distImg = distImg + "Neutral\\" + str(random.randint(1, 90)) + ".jpg"
        
        return distImg
        
        
    # Trial Class
    # There is one instance of this for each trial in the experiment  
    class routineImgData:
        
        def __init__(self):
            self.running = True
            self.imgCounter = 0 
            self.currImg = "Images\\Neutral\\1.jpg"
            self.imgDisplayTimer = core.CountdownTimer(0)
            self.images = list() 
            self.imgDisplayTimes = list()
            self.imagePos = thisExperimentData.imgLocationList[trialCount]
            self.orderImages()
        
        # Ready up images to be shown
        def orderImages(self):
        
            # Determine specific trial's conditions based on global exp data
            
            # Target
            self.targetOrient = thisExperimentData.targetOrientList[trialCount]
            self.targetOffset = thisExperimentData.targetOffsetList[trialCount]
            self.targetPos = random.randint(2 + (self.targetOffset), (totalImageCount - 1) - 2)
            
            # Distractor
            self.distPresent = thisExperimentData.distPresentList[trialCount]
            self.distType = thisExperimentData.distTypeList[trialCount]
            if(self.distPresent):
                self.distPos = self.distPos = self.targetPos - self.targetOffset
            else:
                self.distPos = "NA"
                
            # Add images and display time according to trial conditions
            nonFillerPos = [self.targetPos, self.distPos]
            for i in range(totalImageCount):
                if(i in nonFillerPos):
                    if(i == self.targetPos):
                        img = pickTargetImg(self.targetOrient)
                        self.imgDisplayTimes.append(targetTime)
                    elif(self.distPresent and i == self.distPos):
                        img = pickDistImg(self.distType)
                        self.imgDisplayTimes.append(distractorTime)
                else:
                    img = pickFillerImg()
                    self.imgDisplayTimes.append(fillerTime)
                
                self.images.append(img)
      
        # Handles image change and timer updates
        def displayNextImg(self):
            
    
            if(self.imgCounter >= totalImageCount - 1):
                self.running = False
                return
    
            # set up things for next image
            self.currImg = self.images[self.imgCounter]
            self.imgDisplayTimer.reset()
            self.imgDisplayTimer.addTime(self.imgDisplayTimes[self.imgCounter])
            self.imgCounter += 1
    
    #--- Initialize components for Routine "InputPhase" ---
    KeyInput_InputPhase = keyboard.Keyboard(deviceName='KeyInput_InputPhase')
    Text_InputPhase = visual.TextStim(win=win, name='Text_InputPhase',
        text='What was the orientation of the target image?\n\nPress "LEFT" or "RIGHT"',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "FeedbackPhase" ---
    Text_Feedback = visual.TextStim(win=win, name='Text_Feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "SavingData" ---
    SavingDataText = visual.TextStim(win=win, name='SavingDataText',
        text='Saving Data...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from SaveData
    # create file and path
    absolutePath = os.path.dirname(__file__)
    relativePath = "CustomData"
    dataSavePath = os.path.join(absolutePath, relativePath)
    
    fileName = expInfo['participant'] + "_" + expInfo["session"] + "_Data.xlsx"
    filePath = os.path.join(dataSavePath, fileName)
    
    # --- Initialize components for Routine "ExitScreen" ---
    Text_Ending = visual.TextStim(win=win, name='Text_Ending',
        text='The experiment has ended.\n\nPress "SPACE" to exit the application',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    KeyInput_ExitScreen = keyboard.Keyboard(deviceName='KeyInput_ExitScreen')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "load_functions" ---
    # create an object to store info about Routine load_functions
    load_functions = data.Routine(
        name='load_functions',
        components=[],
    )
    load_functions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for load_functions
    load_functions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    load_functions.tStart = globalClock.getTime(format='float')
    load_functions.status = STARTED
    thisExp.addData('load_functions.started', load_functions.tStart)
    load_functions.maxDuration = None
    # keep track of which components have finished
    load_functionsComponents = load_functions.components
    for thisComponent in load_functions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "load_functions" ---
    load_functions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=load_functions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            load_functions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in load_functions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "load_functions" ---
    for thisComponent in load_functions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for load_functions
    load_functions.tStop = globalClock.getTime(format='float')
    load_functions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('load_functions.stopped', load_functions.tStop)
    thisExp.nextEntry()
    # the Routine "load_functions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    # create an object to store info about Routine WelcomeScreen
    WelcomeScreen = data.Routine(
        name='WelcomeScreen',
        components=[Text_Instructions, KeyInput_Instructions],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for KeyInput_Instructions
    KeyInput_Instructions.keys = []
    KeyInput_Instructions.rt = []
    _KeyInput_Instructions_allKeys = []
    # store start times for WelcomeScreen
    WelcomeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WelcomeScreen.tStart = globalClock.getTime(format='float')
    WelcomeScreen.status = STARTED
    thisExp.addData('WelcomeScreen.started', WelcomeScreen.tStart)
    WelcomeScreen.maxDuration = None
    # keep track of which components have finished
    WelcomeScreenComponents = WelcomeScreen.components
    for thisComponent in WelcomeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    WelcomeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Text_Instructions* updates
        
        # if Text_Instructions is starting this frame...
        if Text_Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Text_Instructions.frameNStart = frameN  # exact frame index
            Text_Instructions.tStart = t  # local t and not account for scr refresh
            Text_Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Text_Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Text_Instructions.started')
            # update status
            Text_Instructions.status = STARTED
            Text_Instructions.setAutoDraw(True)
        
        # if Text_Instructions is active this frame...
        if Text_Instructions.status == STARTED:
            # update params
            pass
        
        # *KeyInput_Instructions* updates
        waitOnFlip = False
        
        # if KeyInput_Instructions is starting this frame...
        if KeyInput_Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            KeyInput_Instructions.frameNStart = frameN  # exact frame index
            KeyInput_Instructions.tStart = t  # local t and not account for scr refresh
            KeyInput_Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(KeyInput_Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'KeyInput_Instructions.started')
            # update status
            KeyInput_Instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(KeyInput_Instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(KeyInput_Instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if KeyInput_Instructions.status == STARTED and not waitOnFlip:
            theseKeys = KeyInput_Instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _KeyInput_Instructions_allKeys.extend(theseKeys)
            if len(_KeyInput_Instructions_allKeys):
                KeyInput_Instructions.keys = _KeyInput_Instructions_allKeys[-1].name  # just the last key pressed
                KeyInput_Instructions.rt = _KeyInput_Instructions_allKeys[-1].rt
                KeyInput_Instructions.duration = _KeyInput_Instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=WelcomeScreen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            WelcomeScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WelcomeScreen
    WelcomeScreen.tStop = globalClock.getTime(format='float')
    WelcomeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WelcomeScreen.stopped', WelcomeScreen.tStop)
    # check responses
    if KeyInput_Instructions.keys in ['', [], None]:  # No response was made
        KeyInput_Instructions.keys = None
    thisExp.addData('KeyInput_Instructions.keys',KeyInput_Instructions.keys)
    if KeyInput_Instructions.keys != None:  # we had a response
        thisExp.addData('KeyInput_Instructions.rt', KeyInput_Instructions.rt)
        thisExp.addData('KeyInput_Instructions.duration', KeyInput_Instructions.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    NoOfTrials = data.TrialHandler2(
        name='NoOfTrials',
        nReps=thisExperimentData.totalTrials, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(NoOfTrials)  # add the loop to the experiment
    thisNoOfTrial = NoOfTrials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisNoOfTrial.rgb)
    if thisNoOfTrial != None:
        for paramName in thisNoOfTrial:
            globals()[paramName] = thisNoOfTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
        
    # Randomly choose one of the two blocks
    block_choice = blockList[trialCount]
    
    #'C': images appear only in the centre in this block
    #'P': images appear only at the periphery (left or right) in this block
    
    for thisNoOfTrial in NoOfTrials:
        curr_block = ''
        NoOfTrials.status = STARTED
        if hasattr(thisNoOfTrial, 'status'):
            thisNoOfTrial.status = STARTED
        currentLoop = NoOfTrials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisNoOfTrial.rgb)
        if thisNoOfTrial != None:
            for paramName in thisNoOfTrial:
                globals()[paramName] = thisNoOfTrial[paramName]
        
        # --- Prepare to start Routine "Fixation" ---
        # create an object to store info about Routine Fixation
        Fixation = data.Routine(
            name='Fixation',
            components=[Fixation_Shape],
        )
        Fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fixation
        Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fixation.tStart = globalClock.getTime(format='float')
        Fixation.status = STARTED
        thisExp.addData('Fixation.started', Fixation.tStart)
        Fixation.maxDuration = None
        # keep track of which components have finished
        FixationComponents = Fixation.components
        for thisComponent in Fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        Eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        print('Starting fixation check...')
        core.wait(0.001)
        callback_mode = 'fixation'
        current_trial_phase = 'fixation'
        
        # --- Run Routine "Fixation" ---
        consecutive_frames_in_fixation = 0
        required_consecutive_frames = 30  # Require 30 consecutive frames (~0.5 seconds at 60fps)
        
        # This loop will display the fixation cross and wait until the gaze_in_fixation flag is True
        while not gaze_in_fixation and consecutive_frames_in_fixation <= required_consecutive_frames:
            # Draw the fixation shape on every frame
            Fixation_Shape.draw()
            win.flip()
            
            # The gaze_data_callback function is running in the background and will
            # update the global gaze_in_fixation flag as soon as the gaze is on the cross.
    
            # Check if gaze has been stable in fixation for required frames
            if gaze_in_fixation:
                consecutive_frames_in_fixation += 1
                print(f"Consecutive frames in fixation: {consecutive_frames_in_fixation}")
                
                # If we have enough consecutive frames, break out of the loop
                if consecutive_frames_in_fixation == required_consecutive_frames:
                    print("Stable fixation achieved!")
                    break
            else:
                consecutive_frames_in_fixation = 0
    
            # Check for the escape key to quit the experiment
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
                return
                

        # --- Ending Routine "Fixation" ---
        # If the while loop above is exited, it means the gaze is on the fixation cross.
        # Now we can end the routine and move to the next part of the trial.
        print('Gaze entered fixation')
        for thisComponent in Fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        # store stop times for Fixation
        Fixation.tStop = globalClock.getTime(format='float')
        thisExp.addData('Fixation.stopped', Fixation.tStop)
        # the Routine "Fixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "ImageDisplay" ---
        # create an object to store info about Routine ImageDisplay
        ImageDisplay = data.Routine(
            name='ImageDisplay',
            components=[ImageComponent],
        )
        ImageDisplay.status = NOT_STARTED
        continueRoutine = True
        
        # update component parameters for each repeat
        # Run 'Begin Routine' code from eyetracker_code
        
        #Switch to image_display mode
        callback_mode = 'image_display'
        current_trial_phase = 'image_display'
        
        # Run 'Begin Routine' code from Code_ImageDisplay
        # Initialise routine data and start running it
        currRoutine = routineImgData()
        
        # store start times for ImageDisplay
        ImageDisplay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ImageDisplay.tStart = globalClock.getTime(format='float')
        ImageDisplay.status = STARTED
        thisExp.addData('ImageDisplay.started', ImageDisplay.tStart)
        ImageDisplay.maxDuration = None
        # keep track of which components have finished
        ImageDisplayComponents = ImageDisplay.components
        for thisComponent in ImageDisplay.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # Update the position attribute of the ImageComponent
        horizontal_offset = win.size[0] / 4
        
        # Create two blocks for image position on screen
        if thisNoOfTrial.thisN <= int(NoOfTrials.nReps / 2):
            if block_choice == 'C':
                ImageComponent.pos = (0, 0)
                curr_block = 'C'
            
            else:
                curr_block = 'P'
                if currRoutine.imagePos == 'LEFT':
                    ImageComponent.pos = (-horizontal_offset, 0)
            
                elif currRoutine.imagePos == 'RIGHT':
                    ImageComponent.pos = (horizontal_offset, 0)
                    
        elif thisNoOfTrial.thisN > int(NoOfTrials.nReps / 2):
            if block_choice == 'C':
                curr_block = 'P'
                if currRoutine.imagePos == 'LEFT':
                    ImageComponent.pos = (-horizontal_offset, 0)
                    
                    
                elif currRoutine.imagePos == 'RIGHT':
                    ImageComponent.pos = (horizontal_offset, 0)
                    
                    
            else:
                ImageComponent.pos = (0, 0)
                curr_block = 'C'
                
        # Start the first image
        currRoutine.displayNextImg()
        ImageComponent.setImage(currRoutine.currImg)
        
        # --- Run Routine "ImageDisplay" ---
        ImageDisplay.forceEnded = routineForceEnded = not continueRoutine
    
        # --- Run Routine "ImageDisplay" ---
        while continueRoutine and currRoutine.running and not end_image_display:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
    
            # Check if the image display timer has expired, and get the next image
            if currRoutine.imgDisplayTimer.getTime() <= 0:
                currRoutine.displayNextImg()
                ImageComponent.setImage(currRoutine.currImg)
                routineTimer.reset()
                
            ImageComponent.draw()
    
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if end_image_display:
                continueRoutine = False
        
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                thisExp=thisExp,
                win=win,
                timers=[routineTimer, globalClock],
                currentRoutine=ImageDisplay,
                )
                continue
    
            # check if all components have finished
            if not continueRoutine:
                ImageDisplay.forceEnded = routineForceEnded = True
                break
    
            # Refresh the screen
            if continueRoutine:
                win.flip()
                
        #Switch back to fixation
        callback_mode = 'fixation'
            
        # --- Ending Routine "ImageDisplay" ---
        for thisComponent in ImageDisplay.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ImageDisplay
        ImageDisplay.tStop = globalClock.getTime(format='float')
        ImageDisplay.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ImageDisplay.stopped', ImageDisplay.tStop)
        # Run 'End Routine' code from eyetracker_code
        thisExp.addData("Null_trial", null_trial)
        if null_trial:
            missed_trials += 1  # Increment missed trials count
        # Run 'End Routine' code from Code_ImageDisplay
        trialCount += 1
        
        # Stop the eyetracker recording until the next trial begins
        Eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        
        # Data for export
        if(currRoutine.distPresent):
            distImg = currRoutine.images[currRoutine.distPos]
            lag = abs(currRoutine.distPos - currRoutine.targetPos)
        else:
            distImg = "NA"
            lag = "NA"
        
        currTrialData = [trialCount, curr_block, currRoutine.images[currRoutine.targetPos], currRoutine.targetPos + 1, currRoutine.targetOrient, \
                         currRoutine.distPos + 1, currRoutine.distType, distImg,lag,\
                         "NOT ANSWERED YET", "NOT ANSWERED YET"]
        # fields = ["Trial No.","Block","Target Image","Target Position","Target Orientation","Distractor Position",
        # "Distractor Type","Distractor Image","Target-Distractor Lag",
        # "Accuracy","Response Time"]
        fullExpData.loc[trialCount - 1] = currTrialData
        # print("Trial " + str(trialCount) + " : " + str(currRoutine.images))
        
        # the Routine "ImageDisplay" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        skip_if_null = data.TrialHandler2(
            name='skip_if_null',
            nReps=0 if null_trial else 1, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(skip_if_null)  # add the loop to the experiment
        thisSkip_if_null = skip_if_null.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSkip_if_null.rgb)
        if thisSkip_if_null != None:
            for paramName in thisSkip_if_null:
                globals()[paramName] = thisSkip_if_null[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisSkip_if_null in skip_if_null:
            skip_if_null.status = STARTED
            if hasattr(thisSkip_if_null, 'status'):
                thisSkip_if_null.status = STARTED
            currentLoop = skip_if_null
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisSkip_if_null.rgb)
            if thisSkip_if_null != None:
                for paramName in thisSkip_if_null:
                    globals()[paramName] = thisSkip_if_null[paramName]
            
            # --- Prepare to start Routine "InputPhase" ---
            # create an object to store info about Routine InputPhase
            InputPhase = data.Routine(
                name='InputPhase',
                components=[KeyInput_InputPhase, Text_InputPhase],
            )
            InputPhase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for KeyInput_InputPhase
            KeyInput_InputPhase.keys = []
            KeyInput_InputPhase.rt = []
            _KeyInput_InputPhase_allKeys = []
            # store start times for InputPhase
            InputPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            InputPhase.tStart = globalClock.getTime(format='float')
            InputPhase.status = STARTED
            thisExp.addData('InputPhase.started', InputPhase.tStart)
            InputPhase.maxDuration = None
            # keep track of which components have finished
            InputPhaseComponents = InputPhase.components
            for thisComponent in InputPhase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            current_trial_phase = 'input'
            
            # --- Run Routine "InputPhase" ---
            InputPhase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisSkip_if_null, 'status') and thisSkip_if_null.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *KeyInput_InputPhase* updates
                waitOnFlip = False
                
                # if KeyInput_InputPhase is starting this frame...
                if KeyInput_InputPhase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    KeyInput_InputPhase.frameNStart = frameN  # exact frame index
                    KeyInput_InputPhase.tStart = t  # local t and not account for scr refresh
                    KeyInput_InputPhase.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(KeyInput_InputPhase, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'KeyInput_InputPhase.started')
                    # update status
                    KeyInput_InputPhase.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(KeyInput_InputPhase.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(KeyInput_InputPhase.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if KeyInput_InputPhase.status == STARTED and not waitOnFlip:
                    theseKeys = KeyInput_InputPhase.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _KeyInput_InputPhase_allKeys.extend(theseKeys)
                    if len(_KeyInput_InputPhase_allKeys):
                        KeyInput_InputPhase.keys = _KeyInput_InputPhase_allKeys[-1].name  # just the last key pressed
                        KeyInput_InputPhase.rt = _KeyInput_InputPhase_allKeys[-1].rt
                        KeyInput_InputPhase.duration = _KeyInput_InputPhase_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *Text_InputPhase* updates
                
                # if Text_InputPhase is starting this frame...
                if Text_InputPhase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Text_InputPhase.frameNStart = frameN  # exact frame index
                    Text_InputPhase.tStart = t  # local t and not account for scr refresh
                    Text_InputPhase.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Text_InputPhase, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Text_InputPhase.started')
                    # update status
                    Text_InputPhase.status = STARTED
                    Text_InputPhase.setAutoDraw(True)
                
                # if Text_InputPhase is active this frame...
                if Text_InputPhase.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=InputPhase,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    InputPhase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in InputPhase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "InputPhase" ---
            for thisComponent in InputPhase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for InputPhase
            InputPhase.tStop = globalClock.getTime(format='float')
            InputPhase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('InputPhase.stopped', InputPhase.tStop)
            # Run 'End Routine' code from Code_InputPhase
            # evaluate answer
            responseTime = KeyInput_InputPhase.rt
            givenAns = KeyInput_InputPhase.keys
            
            leftKeys = ['left']
            leftAns = (currRoutine.targetOrient == "LEFT") and (givenAns in leftKeys)
            
            rightKeys = ['right']
            rightAns = (currRoutine.targetOrient == "RIGHT") and (givenAns in rightKeys)
            if (leftAns or rightAns):
                accuracy = True
                accuracyText = "CORRECT"
            else:
                accuracy = False
                accuracyText = "WRONG"
                
            # update trial data file
            fullExpData.loc[trialCount - 1, "Accuracy"] = int(accuracy)
            fullExpData.loc[trialCount - 1, "Response Time"] = responseTime * 1000 # output in milliseconds
            # check responses
            if KeyInput_InputPhase.keys in ['', [], None]:  # No response was made
                KeyInput_InputPhase.keys = None
            skip_if_null.addData('KeyInput_InputPhase.keys',KeyInput_InputPhase.keys)
            if KeyInput_InputPhase.keys != None:  # we had a response
                skip_if_null.addData('KeyInput_InputPhase.rt', KeyInput_InputPhase.rt)
                skip_if_null.addData('KeyInput_InputPhase.duration', KeyInput_InputPhase.duration)
            # the Routine "InputPhase" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "FeedbackPhase" ---
            # create an object to store info about Routine FeedbackPhase
            FeedbackPhase = data.Routine(
                name='FeedbackPhase',
                components=[Text_Feedback],
            )
            FeedbackPhase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            Text_Feedback.setText(accuracyText)
            # store start times for FeedbackPhase
            FeedbackPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            FeedbackPhase.tStart = globalClock.getTime(format='float')
            FeedbackPhase.status = STARTED
            thisExp.addData('FeedbackPhase.started', FeedbackPhase.tStart)
            FeedbackPhase.maxDuration = None
            # keep track of which components have finished
            FeedbackPhaseComponents = FeedbackPhase.components
            for thisComponent in FeedbackPhase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            current_trial_phase = 'feedback'
            
            # --- Run Routine "FeedbackPhase" ---
            FeedbackPhase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # if trial has changed, end Routine now
                if hasattr(thisSkip_if_null, 'status') and thisSkip_if_null.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Text_Feedback* updates
                
                # if Text_Feedback is starting this frame...
                if Text_Feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Text_Feedback.frameNStart = frameN  # exact frame index
                    Text_Feedback.tStart = t  # local t and not account for scr refresh
                    Text_Feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Text_Feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Text_Feedback.started')
                    # update status
                    Text_Feedback.status = STARTED
                    Text_Feedback.setAutoDraw(True)
                
                # if Text_Feedback is active this frame...
                if Text_Feedback.status == STARTED:
                    # update params
                    pass
                
                # if Text_Feedback is stopping this frame...
                if Text_Feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Text_Feedback.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        Text_Feedback.tStop = t  # not accounting for scr refresh
                        Text_Feedback.tStopRefresh = tThisFlipGlobal  # on global time
                        Text_Feedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Text_Feedback.stopped')
                        # update status
                        Text_Feedback.status = FINISHED
                        Text_Feedback.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=FeedbackPhase,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    FeedbackPhase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in FeedbackPhase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "FeedbackPhase" ---
            for thisComponent in FeedbackPhase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for FeedbackPhase
            FeedbackPhase.tStop = globalClock.getTime(format='float')
            FeedbackPhase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('FeedbackPhase.stopped', FeedbackPhase.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if FeedbackPhase.maxDurationReached:
                routineTimer.addTime(-FeedbackPhase.maxDuration)
            elif FeedbackPhase.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            # mark thisSkip_if_null as finished
            if hasattr(thisSkip_if_null, 'status'):
                thisSkip_if_null.status = FINISHED
            # if awaiting a pause, pause now
            if skip_if_null.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                skip_if_null.status = STARTED
            thisExp.nextEntry()
            
        # completed 0 if null_trial else 1 repeats of 'skip_if_null'
        skip_if_null.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisNoOfTrial as finished
        if hasattr(thisNoOfTrial, 'status'):
            thisNoOfTrial.status = FINISHED
        # if awaiting a pause, pause now
        if NoOfTrials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            NoOfTrials.status = STARTED
        thisExp.nextEntry()
        
    # completed thisExperimentData.totalTrials repeats of 'NoOfTrials'
    NoOfTrials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "SavingData" ---
    # create an object to store info about Routine SavingData
    SavingData = data.Routine(
        name='SavingData',
        components=[SavingDataText],
    )
    SavingData.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for SavingData
    SavingData.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SavingData.tStart = globalClock.getTime(format='float')
    SavingData.status = STARTED
    thisExp.addData('SavingData.started', SavingData.tStart)
    SavingData.maxDuration = None
    # keep track of which components have finished
    SavingDataComponents = SavingData.components
    for thisComponent in SavingData.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SavingData" ---
    SavingData.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.1:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *SavingDataText* updates
        
        # if SavingDataText is starting this frame...
        if SavingDataText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            SavingDataText.frameNStart = frameN  # exact frame index
            SavingDataText.tStart = t  # local t and not account for scr refresh
            SavingDataText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(SavingDataText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'SavingDataText.started')
            # update status
            SavingDataText.status = STARTED
            SavingDataText.setAutoDraw(True)
        
        # if SavingDataText is active this frame...
        if SavingDataText.status == STARTED:
            # update params
            pass
        
        # if SavingDataText is stopping this frame...
        if SavingDataText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > SavingDataText.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                SavingDataText.tStop = t  # not accounting for scr refresh
                SavingDataText.tStopRefresh = tThisFlipGlobal  # on global time
                SavingDataText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'SavingDataText.stopped')
                # update status
                SavingDataText.status = FINISHED
                SavingDataText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SavingData,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            SavingData.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in SavingData.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SavingData" ---
    for thisComponent in SavingData.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SavingData
    SavingData.tStop = globalClock.getTime(format='float')
    SavingData.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SavingData.stopped', SavingData.tStop)
    # Run 'End Routine' code from SaveData
    # save file to pc storage
    dataDF = pd.DataFrame(data = fullExpData, columns = fields)
    with pd.ExcelWriter(filePath) as writer:
        dataDF.to_excel(writer, sheet_name = expInfo['participant'] + "_" + expInfo["session"], index = False)
            
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if SavingData.maxDurationReached:
        routineTimer.addTime(-SavingData.maxDuration)
    elif SavingData.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.100000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "ExitScreen" ---
    # create an object to store info about Routine ExitScreen
    ExitScreen = data.Routine(
        name='ExitScreen',
        components=[Text_Ending, KeyInput_ExitScreen],
    )
    ExitScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for KeyInput_ExitScreen
    KeyInput_ExitScreen.keys = []
    KeyInput_ExitScreen.rt = []
    _KeyInput_ExitScreen_allKeys = []
    # store start times for ExitScreen
    ExitScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ExitScreen.tStart = globalClock.getTime(format='float')
    ExitScreen.status = STARTED
    thisExp.addData('ExitScreen.started', ExitScreen.tStart)
    ExitScreen.maxDuration = None
    # keep track of which components have finished
    ExitScreenComponents = ExitScreen.components
    for thisComponent in ExitScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ExitScreen" ---
    ExitScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Text_Ending* updates
        
        # if Text_Ending is starting this frame...
        if Text_Ending.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Text_Ending.frameNStart = frameN  # exact frame index
            Text_Ending.tStart = t  # local t and not account for scr refresh
            Text_Ending.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Text_Ending, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Text_Ending.started')
            # update status
            Text_Ending.status = STARTED
            Text_Ending.setAutoDraw(True)
        
        # if Text_Ending is active this frame...
        if Text_Ending.status == STARTED:
            # update params
            pass
        
        # *KeyInput_ExitScreen* updates
        waitOnFlip = False
        
        # if KeyInput_ExitScreen is starting this frame...
        if KeyInput_ExitScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            KeyInput_ExitScreen.frameNStart = frameN  # exact frame index
            KeyInput_ExitScreen.tStart = t  # local t and not account for scr refresh
            KeyInput_ExitScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(KeyInput_ExitScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'KeyInput_ExitScreen.started')
            # update status
            KeyInput_ExitScreen.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(KeyInput_ExitScreen.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(KeyInput_ExitScreen.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if KeyInput_ExitScreen.status == STARTED and not waitOnFlip:
            theseKeys = KeyInput_ExitScreen.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _KeyInput_ExitScreen_allKeys.extend(theseKeys)
            if len(_KeyInput_ExitScreen_allKeys):
                KeyInput_ExitScreen.keys = _KeyInput_ExitScreen_allKeys[-1].name  # just the last key pressed
                KeyInput_ExitScreen.rt = _KeyInput_ExitScreen_allKeys[-1].rt
                KeyInput_ExitScreen.duration = _KeyInput_ExitScreen_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ExitScreen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ExitScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ExitScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ExitScreen" ---
    for thisComponent in ExitScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ExitScreen
    ExitScreen.tStop = globalClock.getTime(format='float')
    ExitScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ExitScreen.stopped', ExitScreen.tStop)
    # check responses
    if KeyInput_ExitScreen.keys in ['', [], None]:  # No response was made
        KeyInput_ExitScreen.keys = None
    thisExp.addData('KeyInput_ExitScreen.keys',KeyInput_ExitScreen.keys)
    if KeyInput_ExitScreen.keys != None:  # we had a response
        thisExp.addData('KeyInput_ExitScreen.rt', KeyInput_ExitScreen.rt)
        thisExp.addData('KeyInput_ExitScreen.duration', KeyInput_ExitScreen.duration)
    thisExp.nextEntry()
    # the Routine "ExitScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)

def saveData(thisExp, fullExpData, gaze_data_buffer):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)
    
    # to save the gaze data recorded by the eyetracker
    gaze_output_path = filename + '_gaze.csv'
    if len(gaze_data_buffer) > 0:
        print(f"Saving gaze data to {gaze_output_path}...")
        write_buffer_to_file(gaze_data_buffer, gaze_output_path)
        
    # Create a combined analysis file
    combined_output_path = filename + '_combined.xlsx'
    
    # Convert experimental data to DataFrame if it isn't already
    if isinstance(fullExpData, pd.DataFrame):
        exp_df = fullExpData.copy()
    else:
        exp_df = pd.DataFrame(fullExpData)
    
    #exp_df = exp_df.groupby('Trial No.')
    #exp_df['NoOfTrials.thisN'] = exp_df['NoOfTrials.thisN'] + 1
    
    # Load the gaze data
    if os.path.exists(gaze_output_path):
        gaze_df = pd.read_csv(gaze_output_path)
        
        # Create summary statistics for each trial
        trial_summaries = []
        for trial_num in exp_df['Trial No.']:
            trial_gaze = gaze_df[gaze_df['trial_num'] == trial_num]
            
            if not trial_gaze.empty:
                # Calculate fixation statistics
                fixation_gaze = trial_gaze[trial_gaze['phase'] == 'fixation']
                image_gaze = trial_gaze[trial_gaze['phase'] == 'image_display']
                
                summary = {
                    'Trial No.': gaze_df['trial_num'],
                    'total_gaze_samples': len(trial_gaze),
                    'fixation_samples': len(fixation_gaze),
                    'image_display_samples': len(image_gaze),
                    'valid_gaze_percentage': (trial_gaze['validity'] == 1).mean() * 100,
                    'fixation_duration_ms': len(fixation_gaze) * (1000/60) if len(fixation_gaze) > 0 else 0,  # Assuming 60Hz
                    'image_display_duration_ms': len(image_gaze) * (1000/60) if len(image_gaze) > 0 else 0,
                    'mean_gaze_x': trial_gaze[trial_gaze['validity'] == 1]['gaze_x'].mean() if len(trial_gaze[trial_gaze['validity'] == 1]) > 0 else None,
                    'mean_gaze_y': trial_gaze[trial_gaze['validity'] == 1]['gaze_y'].mean() if len(trial_gaze[trial_gaze['validity'] == 1]) > 0 else None,
                    'gaze_std_x': trial_gaze[trial_gaze['validity'] == 1]['gaze_x'].std() if len(trial_gaze[trial_gaze['validity'] == 1]) > 0 else None,
                    'gaze_std_y': trial_gaze[trial_gaze['validity'] == 1]['gaze_y'].std() if len(trial_gaze[trial_gaze['validity'] == 1]) > 0 else None,
                }
                trial_summaries.append(summary)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(trial_summaries)
        
        # Merge with experimental data
        if not summary_df.empty:
            combined_df = exp_df.merge(summary_df, on = 'Trial_No.', how = 'left')
        else:
            combined_df = exp_df
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(combined_output_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Combined_Data', index=False)
            exp_df.to_excel(writer, sheet_name='Experimental_Data', index=False)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name='Gaze_Summary', index=False)
            
            # Add individual trial sheets for detailed gaze data (optional - for first 5 trials to avoid huge files)
            for trial_num in sorted(exp_df['Trial No.'])[:5]:  # Limit to first 5 trials
                trial_gaze = gaze_df[gaze_df['trial_num'] == trial_num]
                if not trial_gaze.empty:
                    trial_gaze.to_excel(writer, sheet_name=f'Trial_{trial_num}_Gaze', index=False)
        
        print(f"Combined data saved to {combined_output_path}")
    
    print("Data saving complete.")
        
def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process and disconnect eyetracker (if it is still recording)
    if 'Eyetracker' in globals():
        try:
            Eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        except:
            pass
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp,
        win=win,
        globalClock='float'
    )
    saveData(thisExp, fullExpData, gaze_data_buffer)
    quit(thisExp=thisExp, win=win)
