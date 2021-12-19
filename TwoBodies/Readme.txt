

###########################
### Dependencies ###########
###########################
# Python 3 Dependencies:
pip3 install scipy
pip3 install matplotlib
pip3 install numpy

# C++ Dependencies: Qt 5.17

###########################
### Usage: config.ini ###
###########################
# keep parameter step small otherwise integration may not work properly
# keep m1 set to 1 and changed m2 between [0,1]


###########################
### Usage: TwoBodies.exe ###
###########################

## Input data: config.ini
# change config.ini to change simulation

## Output data: data.out

## You can pass path for config.ini as a parameter, if the config.ini is not found default parameters are used
TwoBodies.exe C:/

## Plotting
# plot3d.py is used for plotting, check Usage plot3d.py



###########################
### Usage: twobodies.py ###
###########################

## Input data: config.ini
# otherwise use default parameters defined in code
# keep PARAMETER_step small otherwise integration may not work properly for ex1
# keep m1 set to 1 and changed m2 between [0,1]

## Output data: data.out

# Methods ex0 and ex1 use runge-kutta for integration, ex2 uses ODE from sci framework

# Method with mass ratio and integrating distance between bodies r" and r'
twobodies.py -ex0

# Method where pos1", pos1' and pos2", pos2' are integrated separately
twobodies.py -ex1

# Method using ODE
twobodies.py -ex2


# You can pass path for "config.ini" as second parameter
twobodies.py -ex2 C:/

#########################
### Usage: plot3d.py ####
#########################
# plot data
plot3d.py

# animate data read from "data.out"
plot3d.py -animate

# animate data read from "data.out" with vanishing trail
plot3d.py -animatetrail

# values can be either r(distance between bodies) or x1/x2 positions
# check "data.out" to understand the formatting
