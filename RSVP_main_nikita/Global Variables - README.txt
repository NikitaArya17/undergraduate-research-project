The "GlobalVariables.csv" file is a comma-separated value file that contains the most important variables.

This file must be opened using a notepad application and saved with the same extension of ".csv". 

The code *WILL* break if the file is not saved appropriately

1. TotalTrials
- Determines total number of trials in experiment
- Must be a number divisible by 4

2. TotalImages
- Determines total number of images shown in each trial

3. ImageDisplayTime
- Time that a all images will be shown for (in milliseconds)

4. TargetOffset
- List of lag values target can have in relation to distractor
- must be separated by "_"
- eg. 1_2_8