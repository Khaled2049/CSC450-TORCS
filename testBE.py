#-------------- import here -----------------------------------------
from gym_torcs import TorcsEnv
import numpy as np
#import matplotlib.pyplot as plt
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

HIDDEN1_NODES = 300
HIDDEN2_NODES = 600

# --------------------- define actor and critic model-------------------
def get_actor(state_size,action_space):
  # build your model here
  # Note: we did not compile the model here. In FT2.3, we do not need to compile model
  # if you use eager mode or autograph mode


  x = Input(shape=(state_size,) )   
  h0 = Dense(HIDDEN1_NODES, activation='relu')(x)
  h1 = Dense(HIDDEN2_NODES, activation='relu')(h0)
  Steering = Dense(1,activation='tanh', kernel_initializer=tf.random_normal_initializer(stddev=1e-4))(h1)  
  Acceleration = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1)   
  Brake = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1) 
  V = concatenate([Steering,Acceleration,Brake])          
  model = ks.Model(inputs=x,outputs=V)

  return model

def get_critic(state_size,action_space): #THIS WAS INDENTED FOR SOME REASON??

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

#autograph  # same as update function
@tf.function
def trainmodel(self, state_batch, action_batch, reward_batch, next_state_batch,):
    # here you need to update weights, loss
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )


@tf.function
def train_target(target_weights, weights, tau):
  # this function update target networks weights
  for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# function for everything else
def trainTorcs(train_indicator=0): 
    # if 1 , it will train the model,if 0, it will use train model and run the ai driver
  # --------------------- declare all variables here-------------------------------------
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 5000 #100000
    done = False
    step = 0
    epsilon = 1
    
    hidden_unit1 = HIDDEN1_NODES
    hidden_unit2 = HIDDEN2_NODES
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
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        #s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        s_t = np.hstack((ob.speedX, ob.speedY,  ob.speedZ))
        
        total_reward = 0.
        # TODO: follow code in torcs

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])    
            noise_t = np.zeros([1,action_dim])    
            s_t = tf.expand_dims(tf.convert_to_tensor(s_t, dtype=tf.float32), 0)

            # a_t original predict action with current state
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            # add noise for exploration
            noise_t[0][0] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][0])
            noise_t[0][1] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][1])
            noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][2])

            # The following code do the stochastic brake
            if random.random() <= 0.1:
                noise_t[0][2] = train_indicator * max(epsilon, 0 ) * noise.generate_noise(a_t_original)   

            a_t[0][0] = a_t_original[0][0]  * noise_t[0][0]
            a_t[0][1] = a_t_original[0][1]  * noise_t[0][1]
            a_t[0][2] = a_t_original[0][2]  * noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            #s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            s_t = np.hstack((ob.speedX, ob.speedY,  ob.speedZ))


            s_t1 = tf.convert_to_tensor(s_t1, dtype = tf.float32)

            buff.add(s_t, a_t[0], r_t, s_t1, done)   

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[2] for e in batch])
            

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        target_q_values = target_values(new_states, target_actor, target_critic)

        # Discounted Qvalues
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]

        # updating target actor and target critic
        if (train_indicator):
            loss += update(actor_model, critic_model, states, y_t, actor_optimizer, critic_optimizer)
            update_target(target_actor.variables, actor_model.variables, TAU)
            update_target(target_critic.variables, actor_model.variables, TAU)
        
        total_reward += r_t
        s_t = s_t1
    
        if np.mod(j,100) == 0: 
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
        step += 1
        if done:
            break

    if np.mod(i, 3) == 0:
        if (train_indicator):
            print("Now we save model")
            actor.model.save_weights("actormodel.h5", overwrite=True)
            with open("actormodel.json", "w") as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights("criticmodel.h5", overwrite=True)
            with open("criticmodel.json", "w") as outfile:
                json.dump(critic.model.to_json(), outfile)

    print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")
        



if __name__ == "__main__":
    trainTorcs(1)
