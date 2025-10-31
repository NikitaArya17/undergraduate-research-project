# Undergraduate Psychology Research Project
For this project, I contributed to a codebase maintained by the Brain and Cognition Laboratory of Ahmedabad University, Ahmedabad, Gujarat, India, which is used to run 
run psychology experiments that study human attention and visual processing of information. The code has been written in Python and extensively uses the Psychopy library and its associated packages. Psychopy is designed especially to conduct experiments involving the presentation of auditory and visual stimuli, and to record participants' responses, which may be made via mouse, keyboard or other externally connected device.

It should be noted that this codebase has resulted from the work that several students have done before I joined the project.

My specific contributions include the section of code involved in integrating the existing code with an externally connected [Tobii Pro Nano](https://connect.tobii.com/s/products/more/discontinued/tobii-pro-nano?language=en_US) eye-tracking device, so that the program could collect gaze data recorded by the device, and use this input to decice when to present a series of images to participants.

My contributions in this area include modifications to the `gaze_data_buffer()` function, the addition of eye-tracker calibration section in the `run()` function and a modification of the 'Fixation routine' in the `run()` function by adding a flag that will evaluate to `True` and produce the required output when the participant directs their gaze to the correct area on the screen.

The latest version of the Psychopy software must be installed to run this code. Due to its specialised nature, the code cannot be run on a standard interpreter or IDE.
