'''
file:       project.py (Untitled0.ipynb on Google Colaboratoty)
context:    TORCS project, group 1 in CSC450, taught by Dr. Iqbal. The found-
            ation of this code was provided by Mr. Islam.
project collaborators:  Blake Engelbrecht
                        David Engleman
                        Shannon Groth
                        Khaled Hossain
                        Jacob Rader
'''

#-------------- import here -----------------------------------------
from gym_torcs import TorcsEnv
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as ks
from tensorflow.keras.layers import Input,Dense,concatenate,add
from tensorflow.python.framework.ops import disable_eager_execution
# importing classes
from ReplayBuffer import ReplayBuffer
from OUActionNoise import OUActionNoise
import json

# --------------------- define actor and critic model-------------------
def get_actor(state_size):
    # build your model here
    # Note: we did not compile the model here. In FT2.3, we do not need to compile model
    # if you use eager mode or autograph mode
    
    x = Input(shape=(state_size,))   
    h0 = Dense(HIDDEN1_NODES, activation='relu')(x)
    h1 = Dense(HIDDEN2_NODES, activation='relu')(h0)
    Steering = Dense(
        1,
        activation='tanh',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-4)
        )(h1)  
    Acceleration = Dense(
        1,
        activation='sigmoid', 
        kernel_initializer=tf.random_normal_initializer(stddev=1e-4)
        )(h1)   
    Brake = Dense(
        1,
        activation='sigmoid',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-4)
        )(h1) 
    V = concatenate([Steering,Acceleration,Brake])          
    model = ks.Model(inputs=x,outputs=V)
    
    return model

def get_critic(state_size,action_space):
    # you don't either need to compile model here
    # just build the model
    S = Input(shape=(state_size,))  
    A = Input(shape=(action_space,),name='action2')
    
    w1 = Dense(HIDDEN1_NODES, activation='relu')(S)
    a1 = Dense(HIDDEN2_NODES, activation='linear')(A) 
    
    h1 = Dense(HIDDEN2_NODES, activation='linear')(w1)
    h2 = add([h1,a1])    
    h3 = Dense(HIDDEN2_NODES, activation='relu')(h2)
    
    V = Dense(3,activation='linear')(h3)   
    model = ks.Model(inputs=[S,A],outputs=V)
    
    return model

#autograph
@tf.function
def trainmodel(): # same as update function
    pass
    # here you need to update weights, loss
    
    
@tf.function
def train_target(target_weights, weights, tau):
    # this function update target networks weights
    for (a, b) in zip(target_weights, weights):
          a.assign(b * tau + a * (1 - tau))
    
# function for everything else
def trainTorcs(train_indicator=1): # if 1 , it will train the model,if 0, it will use train model and run the ai driver
    # --------------------- declare all variables here-------------------------------------
    
    #-----------------------Create buffer here-----------------------
    
    # TODO: add noise 
    
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer, using Replay buffer class
    # create model for actor and critic
    actor_model  = get_actor(hidden_unit1, hidden_unit2)
    critic_model = get_critic(hidden_unit1, hidden_unit2)
    
    #actor_model.summary()
    #critic_model.summary()
    
    # create target actor and critic
    target_actor = get_actor(hidden_unit1, hidden_unit2)
    target_critic = get_critic(hidden_unit1, hidden_unit2)
    
    # initialize target weights same as acot and critic (default) weights
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    
    # set optimizer
    critic_optimizer = tf.keras.optimizers.Adam(LRC)
    actor_optimizer = tf.keras.optimizers.Adam(LRA)
    
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    
    #----------------initially we don't have save weights, but once we have saved weights. we load those weights.---------------------
    #Now load the weight
    print("Now we load the weight")
    try:
        actor_model.load_weights("actormodel.h5")
        critic_model.load_weights("criticmodel.h5")
        target_actor.load_weights("actormodel.h5")
        target_critic.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
      
        #--------------------- from here we will use loops to tain model---------------------------
      
        for i in range(max_episode):

            # TODO: follow code in torcs
        
            for j in range(max_steps):
                pass
                # TODO: get state
                
                # TODO: generate action
                # tf.convert_to_tensor(states, dtype=tf.float32)
                #actions = actor_model(states)
                  
                # TODO: add noise to actions
                  
                
                
                # ob, r_t, done, info = env.step(a_t[0])
                  
                # TODO: save it to buffer
                  
                  
                #------------------------ get states actions from buffer---------------------------
                  
                #TODO tain model --- call trainmodel
                  
                  
                # rest will be same as in DDPG




if __name__ == "__main__":
    trainTorcs(1)























