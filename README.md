# Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Quick Run
---

* Install dependencies
* [Download model](https://drive.google.com/file/d/1wOHHasGP3zMBEGJmsXD9jYiH-7hTVMwU/view?usp=sharing).
* Run

```sh
python drive.py model.h5
```

and then start the simulator in *autonomous* mode ([download simulator]( https://github.com/udacity/self-driving-car-sim )).

![alt text](examples/sim_image.png)

Overview
---

This project is aimed at training a neural network to drive a car autonomously inside a simulator through behavioral cloning. 

The car is first driven manually, and the output of the controls (Steering Angle, Throttle, Break and Speed) are recorded into a .csv file, as well as the image being seen by the car from 3 different viewports (center, left shield and right shield). This is then fed into a neural network that learns the desired output controls for the current image observed. 

The files used in the project:

* **Simulator** : The executable version for Windows, Linux and Mac can be found in this [repository]( https://github.com/udacity/self-driving-car-sim ). 
* *Model.ipynb* : Contains the network model architecture. 
* *model.h5* : The network architecture and weights, used by the simulator to run the car autonomously. 
* *drive.py* : Used to run the simulation using the trained model. 
* *video.mp4* : A sample result

Dependencies
---

Ideally a GPU is needed to run the model in realtime. Python 3.6 is used with keras 2.2, and a tensorflow backend. Other dependencies for simulation include: socketio, eventlet, flask. 

Training the Network
---

### Recording the data

1. Download and run the simulator
2. "Play"
3. "Training Mode"
4. Press **R**
5. (Create and) select *data/* folder (relative to repository root).
6. Press **R** again to start recording. 
7. Drive around using **W** and hold-and-drag the mouse button to steer the wheel. 
8. Press **R** again to stop recording. 

### Run the Network

1. Run all the cells inside *Model.ipynb*, and the model will automatically be saved as model.h5. 

***Addiotional notes***

The directory separator character might vary from system to system. On windows it is the backslash, '\', but on Linux it is usually forward slash. This affects the third cell from the notebook, when reading the data, and will cause errors if not set correctly. 

Training the network for successful operation of the car will usually require a large amount of driving recording, and specific cases of drive recovery (when the car deviates from the route and acts to come back to center). 
