The context of this software is to meet project requirements of project 1 (An AI Driver for TORCS)
in CSC450-001 FA2020 taught by Dr. Razib Iqbal. An Environment installation guide is included in
this readme, as well as contact information for the team members that have collaborated on this project. 
-------------------------------------------------
----- Environment installation guide -------
-------------------------------------------------

-------------- 1. Requirements: --------------

** Ubuntu >= 16.04
** OpenGL 1.3
** GPU that is compatible with CUDA (typically NVIDIA graphics cards)
** CUDA (https://developer.nvidia.com/cuda-toolkit)
** Anaconda (https://www.anaconda.com/products/individual)
** Python 3.7
** Spyder IDE (https://www.spyder-ide.org/)

------------- 2. Terminal Entry: -------------

$ sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 
libxcomposite1 libasound2 libxi6 libxtst6
<make sure Anaconda is part of the system path variable with 'export PATH="/home/username/
anaconda3/bin:$PATH" for the conda command to work>
$ conda env create –-name turinglab python=3.7
$ cd anaconda3/envs/turinglab
$ sudo apt install git 
$ git clone https://github.com/openai/gym.git
$ cd gym
$ sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg 
xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
$ pip install gym
$ sudo apt-get install xautomation
$ git clone https://github.com/ugo-nama-kun/gym_torcs.git
$ cd gym_torcs
$ cd vtorcs-RL-color
$ sudo apt-get install libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libplib-dev 
libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev libpng-dev
$ ./configure
$ make
<this may result in a error that needs to be fixed: inside geometry.cpp (in /home/username/
anaconda3/envs/turinglab/gym/gym_torcs/vtorcs-RL-color/src/drivers/olethros) line 373 where
"isnan(r)" must be replaced with "std::isnan(r)" after saving this change, execute 'make' again>
$ sudo make install
$ sudo make datainstall
<we can now launch TORCS with 'sudo torcs' in the terminal>

------- 3. How to execute this project ------

This project comes with a model that was saved after ~9 hours of training. To use the model
that was trained change "trainTorcs(1)" to "trainTorcs(0)" at the end of the file in ddpg.py; the 
integer that is passed in the "trainTorcs" function call is the train_indicator value, passing 1 means
the model is in training mode and passing 0 means the model uses the last-saved weights
(actormodel.h5 & criticmodel.h5 in the Final Code folder). 

Before we launch the Spyder IDE, activate the turinglab Anaconda environment so we can utilize
all of the libraries that we installed earlier:
$ conda activate turinglab
<you should see "(turinglab)" before the input line in the terminal when it's activated>
$ spyder
In the Spyder IDE open "ddpg.py" located in the Final Code folder, run the program with the run
button in the editor toolbar to run the file within the Spyder IDE.

-------------------------------------------------
-------- Team Contact Information ---------
-------------------------------------------------
Blake Engelbrecht                                          
Department of Computer Science                    
Missouri State University                                           
engelbrecht95@live.missouristate.edu           
-------------------------------------------------
David Englemen  
Department of Computer Science 
Missouri State University 
englemen369@live.missouristate.edu 
-------------------------------------------------
Shannon Groth                                                 
Department of Computer Science                    
Missouri State University                                 
campbell489@live.missouristate.edu                 
-------------------------------------------------
Khaled Hossain
Department of Computer Science 
Missouri State University 
khaled2049@live.missouristate.edu 
-------------------------------------------------
Jacob Rader 
Department of Computer Science 
Software Development Major
Missouri State University
rader503@live.missouristate.edu 
-------------------------------------------------