'''
file:       ddpg.py
context:    TORCS project, team 1 in CSC450, taught by Dr. Razib Iqbal. 
            Main File
            
project collaborators:  Blake Engelbrecht
                        David Engleman
                        Shannon Groth
                        Khaled Hossain
                        Jacob Rader

LICENSE INFORMATION:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Cross-references material is denoted with: ### <reference> ###
'''
### Functional Requirement 7 is represented by the first 5 class
# methods in this source code. These bits of code utilize the 
# TensorFlow 2.0 library to create a deep learning AI bot. ###

#-- Importing Classes ----------------------------------------
from gym_torcs import TorcsEnv
from ReplayBuffer import ReplayBuffer
from OUActionNoise import OUActionNoise
#-- Import Libraries -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as ks
from tensorflow.keras.layers import Input,Dense,concatenate,add
from tensorflow.python.framework.ops import disable_eager_execution
import json
#-- Set global variables -------------------------------------
noise = OUActionNoise()
HIDDEN1_NODES = 300
HIDDEN2_NODES = 600
output_filename = "Log_Output.txt"

# -- Define Actor Model --------------------------------------
def get_actor(state_size,action_space):
    '''
    Return actor model
    '''
    x = Input(shape=(state_size,) )   
    h0 = Dense(HIDDEN1_NODES, activation='relu')(x)
    h1 = Dense(HIDDEN2_NODES, activation='relu')(h0)
    Steering = Dense(1,activation='tanh', kernel_initializer=tf.random_normal_initializer(stddev=1e-4))(h1)  
    Acceleration = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1)   
    Brake = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1) 
    V = concatenate([Steering,Acceleration,Brake])          
    model = ks.Model(inputs=x,outputs=V)
    return model

#-- Define Critic Model --------------------------------------
def get_critic(state_size,action_space):
    '''
    Return critic model
    '''
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

#-- Autograph ------------------------------------------------
@tf.function
def update(actor_model,critic_model,states,actions,y,actor_optimizer,critic_optimizer):
    ''' 
    Training and updating Actor & Critic networks
    '''
    y = tf.cast(y, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        critic_value = critic_model([states, actions], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(states, training=True)
        critic_value = critic_model([states, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )
    return critic_loss


@tf.function
def target_values(new_states, target_actor,target_critic):
    '''
    Calculate target values
    '''
    # target action for batch size new_states
    target_actions = target_actor(new_states)
    
    # target Qvalue for the batch size
    target_q_values = target_critic([new_states,target_actions ])  
    return target_q_values

@tf.function
def train_target(target_weights, weights, tau):
    '''
    Update target network weights
    '''
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

#-- Main Function --------------------------------------------
def trainTorcs(train_indicator=1): 
    '''
    train_indicator = 1 to train the model
    train_indicator = 0 to use the model
    '''
    #-- Declare local variables ------------------------------
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Learning rate for Critic

    action_dim = 3  #Steering=1, Acceleration=2, Brake=3
    state_dim = 29  #Number of sensor input

    np.random.seed(1337) 

    vision = False  

    EXPLORE = 100000.
    episode_count = 200
    max_steps = 10000 
    done = False
    step = 0
    epsilon = 1
    
    FRESH_START = False     #Start without using saved model
                            #AND overwrite model!!!!
    actor_save_file = "actormodel.h5"       #Actor model file
    critic_save_file = "criticmodel.h5"     #Critic model file
    output_filename = "Log_Output.txt"      #Log output file
    
    #-- Create ReplayBuffer ----------------------------------
    buff = ReplayBuffer(BUFFER_SIZE)
    #-- Create actor model and critic model ------------------
    actor_model  = get_actor(state_dim, action_dim)
    critic_model = get_critic(state_dim, action_dim)
    #-- Get target models ------------------------------------
    target_actor = get_actor(state_dim, action_dim)
    target_critic = get_critic(state_dim, action_dim)
    #-- Init default weights for actor and critic models -----
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    #-- Set optimizer for both models ------------------------
    critic_optimizer = tf.keras.optimizers.Adam(LRC)
    actor_optimizer = tf.keras.optimizers.Adam(LRA)

    #-- Generate TORCS environment from gym_torcs.py ---------
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    ### Creating a TORCS environment represents Functional 
    # Requirement 1 & Non-functional requirement 7.
    # The TorcsEnv class in gym_torcs.py uses
    # snakeoil3_gym to create the UDP connection. The client-
    # server architecture diagram is represented here as well. ###

    ### Raw sensor data is read in gym_torcs.py; creating 
    # the TORCS environment using gym_torcs.py represents 
    # Funtional Requirement 5. ###

    ### Non-functional requirement 6 is represented here as 
    # the TORCS environment is restricted to data 
    # transmissions at a rate of 20ms. ###
    
    # Compare max_reward to total_reward each episode, if
    # total_reward is greater, update max_reward and save
    # weights
    max_reward = -1 

    #-- Set up file stream for log ouput ---------------------
    file_output_str = "\n\n##########Starting New Experiment##########\n" 
    log_text_file = open(output_filename, "a")
    log_text_file.write(file_output_str)
    file_output_str = ""

    #-- Load weights -----------------------------------------
    print("Now we load the weight")
    if FRESH_START==False:
        try:
            actor_model.load_weights(actor_save_file)
            critic_model.load_weights(critic_save_file)
            target_actor.load_weights(actor_save_file)
            target_critic.load_weights(critic_save_file)
            print("Weight load successfully")
            file_output_str = "\nWeight load successfully"
        except:
            print("Cannot find the weight")
            file_output_str = "\nCannot find the weight"
    else:
        print("Skipping weight load...")
        file_output_str = "\nSkipping weight load..."
    log_text_file.write(file_output_str)

    #-- Train model ------------------------------------------
    #To store reward history of each episode
    ep_reward_list = []
    #To store average reward history of last few episodes
    avg_reward_list = []
    
    print("TORCS Experiment Start.")
    file_output_str = "\nTORCS Experiment Start."
    log_text_file.write(file_output_str)
    #Begin episode loop...
    for i in range(episode_count):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        file_output_str = "\nEpisode : " + str(i) + " Replay Buffer " + str(buff.count())
        log_text_file.write(file_output_str)
        
        #Relaunch TORCS every 3 episode because of the memory leak bug
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   
        else:
            ob = env.reset()
        
        #Create horizontal stack of current input variables
        s_t = np.hstack((
            ob.angle, 
            ob.track, 
            ob.trackPos, 
            ob.speedX, 
            ob.speedY, 
            ob.speedZ, 
            ob.wheelSpinVel/100.0,
            ob.rpm))
        
        total_reward = 0.

        #Begin step loop...
        for j in range(max_steps):
            
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            #Convert horizontal stack to tensor for optimization
            s_t = tf.expand_dims(tf.convert_to_tensor(s_t, dtype=tf.float32),0)
            
            #Predict action with current state
            a_t_original = actor_model(s_t)
            
            #Generate noise for exploration
            noise_t[0][0] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][2], -0.1 , 1.00, 0.05)
            #Add noise to action
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            
            
            #Get next step with predicted action and next observation
            ob, r_t, done, info = env.step(a_t[0])
            
            #Create horizontal step for the next state
            s_t1 = np.hstack((
                ob.angle,
                ob.track,
                ob.trackPos,
                ob.speedX,
                ob.speedY,
                ob.speedZ,
                ob.wheelSpinVel/100.0,
                ob.rpm))
            
            #Convert horizontal stack to tensor for optimization
            s_t1 = tf.convert_to_tensor(s_t1,dtype=tf.float32)

            #Record current_state, action, reward, next_state
            buff.add(s_t[0], a_t[0], r_t, s_t1, done)      
            
            #Update batch with replay buffer
            batch = buff.getBatch(BATCH_SIZE) 
            
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
            
            target_q_values = target_values(new_states,target_actor,target_critic) 
            #Discounted Qvalues
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
            
            if (train_indicator):
                loss += update(actor_model,critic_model,states,actions,y_t,actor_optimizer,critic_optimizer)
                
                # update target actor and target critic
                train_target(target_actor.variables, actor_model.variables, TAU)
                train_target(target_critic.variables, critic_model.variables, TAU)

                ### Training the target uses a method of calculation (TAU) that keeps 
                # the trackPos value between -1 and 1, this represents Functional
                # Requirement 2. Functional Requirement 9 is also satisfied here, by the same
                # means as FR 2 is satsfied (through the trackPos calculation). 
                # Non-function Requirement 4 is handled by this calculation as well;
                # with a longer training experiment, the trackPos value is tuned more
                # smoothly. ###

                ### Training the model to its target values also represents the 
                # TORCS training architecture and the car control diagrams in the
                # SDD. ###
            
            #Increase total_reward for episode
            total_reward += r_t
            #Advance old state to new state
            s_t = s_t1
        
            #Output step results to log
            print("Episode", i, "Step", step, "Reward: %.3f"%r_t, "Loss: %.3f"%loss)
            file_output_str = "\nEpisode " + str(i) + " Reward " + str(r_t) + " Loss: " + str(loss)
            log_text_file.write(file_output_str)

            step += 1
            #If episode is done, move on to next epsiode
            if done:
                break
            
        #Added this for the graph
        ep_reward_list.append(total_reward)

        #Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(j, avg_reward))
        avg_reward_list.append(avg_reward)
        
        #Save model if total_reward is greater than max_reward...
        print("Now we save model if total_reward > max_reward:")  
        file_output_str = "\nNow we save model if total_reward > max_reward:"
        log_text_file.write(file_output_str)
        if (total_reward > max_reward):
            max_reward = total_reward
            print("New Max Reward = ", max_reward)
            file_output_str = "\nNew Nax Reward = " + str(max_reward)
            log_text_file.write(file_output_str)
            print("Saving weights...")
            file_output_str = "\nSaving weights..."
            log_text_file.write(file_output_str)
            
            actor_model.save_weights(actor_save_file, overwrite=True)
            with open("actormodel.json", "w") as outfile:
                json.dump(actor_model.to_json(), outfile)

            critic_model.save_weights(critic_save_file, overwrite=True)
            with open("criticmodel.json", "w") as outfile:
                json.dump(critic_model.to_json(), outfile)
        else:
                print("Reward not great enough...")
                file_output_str = "\nReward not great enough..."
                log_text_file.write(file_output_str)
                print("total_reward =", total_reward, " vs. max_reward =", max_reward)
                file_output_str = "total_reward = " + str(total_reward) + " vs. max_reward = " + str(max_reward)
                log_text_file.write(file_output_str)
        
        #Output episode results to log
        print()
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        file_output_str = "\nTOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward)
        log_text_file.write(file_output_str)
        print("Total Step: " + str(step))
        file_output_str = "\nTotal Step: " + str(step) 
        log_text_file.write(file_output_str)
        print()
        
    #This is for shutting down TORCS at the end of experiment
    env.end()  
    print("Finish.")
    file_output_str = "\nFinish.\n"
    log_text_file.write(file_output_str)
    file_output_str = "\n----------TO USE THIS WEIGHT SET TRAIN INDICATOR TO 0----------\n"
    log_text_file.write(file_output_str)
    log_text_file.close()
    
    #Plotting graph
    #Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
        


if __name__ == "__main__":
    #Pass 1 for training the model, or 0 for using the model
    trainTorcs(1)