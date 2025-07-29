# Occultation of a star by an asteroid as a method of detection of its rings
**Learning goal**: To investigate the possibility of asteroid rings detection in different configurations from direct images of stellar occultation.

**Product goal**: To create a series of programs, carrying out calculations of system parameters and its synthetic observations at the stage of occultation.

## Tasks
1) Create numerous models of ring systems with different orbital and physical parameters
2) Model the process of a star occultation by an asteroid with a ring system to construct the light curve of the event

## Contributors
Author: D.E.Veliev

Supervisor: S.N.Kolyakina

Subject advisors: E.O.Dedov, A.A.Vakhonin

## Project prospects

### Future use of the project
My programs and synthetic lightcurve of the event will help professional and hobby astronomers to identify an asteroid with rings when analyzing a section of the sky.

### Relevance
Nowadays, there is a large number of observational missions (ZTF, PanSTARRS and others) that provide a vast amount of data to the public. This data can be used, among other things, to explore small bodies of the Solar System. In this work, modeling of asteroid ring systems is carried out in order to investigate the possibility of their detection from direct images. The models can be used to identify ring systems using data from various observational missions, which is an urgent task in the field of work with a large amount of astronomical data.
Asteroids with satellites or rings are rarely observed due to the difficulties associated with the registration of such objects. The study of the evolution of such systems is extremely important for understanding the formation processes not only of such asteroids, but also of planetary systems as a whole.
The proposed method of testing the possibility of detecting asteroids with rings based on stellar occultations can be used to detect new objects, including near-Earth asteroids, using real data.

## Usage
### What the application does

1) Calculates all the parameters needed for the occultations
2) Builds a predicted lightcurve for each set of the parameter values
3) Allows to compare observational data with the model

### How to run and use
1) Install the required modules (requirements.txt)
2) Run the main.py program
3) Select the parameters ranges for the model in the first widget - you can either click "Auto" to set a parameter range to default or click "Manual" to manually set the minimum and the maximum values of the range
4) Load a CSV observations file for compairing any observations with the model (not required, you can proceed without an observations file)
5) Select the parameter values in the new window to view the predicted occultation lightcurve for the bodies with the selected physical properties (wait if the window is not responding, calculations take some time)
6) Move sliders below the graph for aligning the observations data with the graph
7) Click "Simulate Occultation Animation" if you want to see the occulation visualization for a certain set of the parameters

## Programs

formulas.py - used for storing the formulas used in the calculations

main.py - used to run the model

measure.py - contains a specific class for working with parameter ranges and their units of measurement

models.py - creates NumPy representations for all the objects used in the model, realizes the method of the occultation

observations.py - constains the class for loading and working with the observation data

space.py - contains the main celestial body classes used for the model (including the asteroid, its rings and the star models)

units.py - contains the units of measurement used in the project

visualization - creates the parameter selection, lightcurve and animation widgets
