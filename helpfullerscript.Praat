# This script extracts the mean and max intensity of an audio file
# and the mean and max value of each of the first two formants.
# It then writes these values out to a file

# Give Praat the path to the directory where your audio files are stored
# ASSIGN THIS VARIABLE TO THE PATH TO YOUR AUDIO DATA DIRECTORY
directory$ = "C:/Users/acm75/OneDrive/Desktop/PSYC684_PS1/Q5/Dialects/"

# Give Praat the path to your output file
# ASSIGN THIS VARIABLE TO THE FULL PATH TO YOUR OUTPUTFILE
outputfile$ = "C:/Users/acm75/OneDrive/Desktop/PSYC684_PS1/Q5/dialectsdataoutput.csv"
deleteFile: outputfile$ 

# print a line to your output file containing the names of the features
appendFileLine: outputfile$, "minint,maxint,rangeint,meanint,pitch,maxf1,meanf1,maxf2,meanf2,dx"

# loop through the files one by one
ctr = 0
strings = Create Strings as file list: "list", directory$ + "/*.mp3"
numberOfFiles = Get number of strings
for ifile to numberOfFiles
	ctr = ctr + 1
    selectObject: strings
    fileName$ = Get string: ifile
    appendInfoLine: fileName$

    # this is just so you know the name of the sound you're dealing with
    # that way, you can select it later on if you need to
    fileIntro$ = replace$ (fileName$, ".mp3", "", 1)
    appendInfoLine: fileIntro$

    # read in the sound file
    Read from file: directory$ + "/" + fileName$
    appendInfoLine: fileIntro$ + "read!"

    # make sure you've selected the sound
    selectObject: "Sound " + fileIntro$

    # get min, max and mean intensity (a.k.a. energy) of that sound
    # also get intensity range
    To Intensity: 100, 0, "yes"
    appendInfoLine: fileIntro$ + "intensity!"

    minintensity = Get minimum: 0, 0, "Parabolic"
    maxintensity = Get maximum: 0, 0, "Parabolic"
    meanintensity = Get mean: 0, 0, "energy"
    rangeintensity = maxintensity - minintensity

    # print the info out to a file
    appendFile: outputfile$, minintensity, ",", maxintensity, ",", rangeintensity, ",", meanintensity, ","

    # make sure you select the sound again
    selectObject: "Sound " + fileIntro$

    # convert to pitch
    To Pitch: 0, 75, 500
    appendInfoLine: fileIntro$ + " pitch!"
    # select it
    #select Pitch: "Pitch " + fileIntro$

    # get mean F0
    # maxF0  = Get maximum: 
    meanF0 = Get mean: 0, 0, "Hertz"

    # convert to pointProcess for shimmer
    To PointProcess
    pp = selected ("PointProcess")

    # print to file
    appendFile: outputfile$, meanF0, ","
 

    # make sure you select the sound again
    selectObject: "Sound " + fileIntro$

    # get formant information for that sound
    To Formant (burg): 0, 5, 5500, 0.025, 50
    appendInfoLine: fileIntro$ + " formant!"

    for formnum from 1 to 2
			 appendInfoLine: "formant"
	 # minf = Get minimum: formnum, 0, 0, "Hertz", "Parabolic"
         maxf = Get maximum: formnum, 0, 0, "Hertz", "Parabolic"
         # rangef = maxf - minf
         meanf = Get mean: formnum, 0, 0, "Hertz"
	 		 appendFile: outputfile$, maxf, ",", meanf, ","
    endfor

  
    appendInfoLine: "data complete!"
	appendInfoLine: ctr

    # finally, write out the class (northern or southern)
	  ## IF YOU WANT TO USE WEKA COMMENT THIS OUT
    if startsWith(fileIntro$, "southern")
	     appendFileLine: outputfile$, "1"        
    else
	     appendFileLine: outputfile$, "0"        
    endif

	  ## IF YOU WANT TO USE WEKA UNCOMMENT
    #if startsWith(fileIntro$, "southern")
	  #   appendFileLine: outputfile$, "south"        
    #else
	  #    appendFileLine: outputfile$, "north"        
    #endif


  	 removeObject: "Sound " + fileIntro$
	 removeObject: "Formant " + fileIntro$
	 removeObject: "Intensity " + fileIntro$
	 removeObject: "Pitch " + fileIntro$
	 

endfor


removeObject: "Strings list"