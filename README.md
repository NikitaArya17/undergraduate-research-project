# Undergraduate Psychology Research Project
In this project, I contributed to a codebase that has been developed and maintained by several students of the Brain and Cognition Laboratory of Ahmedabad University.

This code is used to run cognitive psychology and neuroscience experiments that study, among other things, human attention and visual processing of information.

The code has been written in Python and extensively uses the PsychoPy library and its associated packages. PsychoPy is designed specifically to conduct experiments involving the presentation of auditory and visual stimuli, and to record participants' responses, which may be made via mouse, keyboard or other externally connected devices.

My work involved refactoring the section of code responsible for integrating the existing code with an externally connected [Tobii Pro Nano](https://connect.tobii.com/s/products/more/discontinued/tobii-pro-nano?language=en_US) eye-tracking device, which records the exact position at which a participant directs their gaze on a computer screen.

The program evaluates this input and presents a series of visual stimuli to participants as soon as they focus their gaze on the centre of the screen.

My contributions include modifications to the `gaze_data_buffer()` function, the addition of an eye-tracker calibration section in the `run()` function and a modification of the 'Fixation routine' in the `run()` function by adding a flag that will evaluate to `True` and produce the required output when the participant directs their gaze to the correct area on the screen.

The latest version of the PsychoPy software must be installed to run this code. Due to its specialised nature, the code cannot be run on a standard interpreter or IDE.

## Potential Applications in Digital Health ##
### Diagnostic Assistance for Attention-Related Disorders
As this program precisely records human eye movements and gaze data, it could potentially be used as a diagnostic tool for attention-related conditions such as ADHD, though more research would be required in order to improve its accuracy and suitability for medical purposes.
