# Reinforcement Learning using T3D algorithm

Goal here is to get the car to stay on road using Reinforcement Learning - T3D algorithm. 
I have taken a phased approach to this problem
 >- Phase 1 - Integrate T3D learning to the Kivy environment ; with sensor data using image patch brightness ( converting to scalar ) as the state dimension. Fully connected Actor and Critic models were used.Here idea was a get a understanding of the working of TD3. 

 >- Phase 2 - Replace the fully connected network in the above network with a CNN and feed the the image frames in front of the car(took 3 at various angles) as sensory data to it.

Describing below both the phases

## Phase 1 
  In this phase i have taken car environment and integrated it with the T3D learning. I have NOT used CNN in this phase.

  https://youtu.be/6NWjU9S8zro

  ### Parameters used
  >- **Action Dimension** : 1 - angle of rotation
  >- **State Dimension** : 5
  >>- 3 brightness parameter from sensor patches
  >>- 2 oreintations from target x,y coordinates
  >- Timesteps of taking action : 1000
  >- Episode **done** if
  >>- Max epsidoes steps reached : 1200
  >>- Car gets to the edges
  >>- Reaches the destination
  >- **Rewards**
  >>- On sand = - 1
  >>- Car hits the edges = - 1
  >>- If on Road and distance is reducing = + 1
  >>- On Road = + 0.8

  ### Network

  Used fully connected layers for Actor and Critic Model

  ### Observations
  Started driving on the road after 27 episodes




## Phase 2

  In this phase integrated it with the T3D learning and passed the sensory data as image frames to a CNN
  
  >- Attempt 1 : With 3 patches in fron of car as sensory data only as input  
     >>- https://youtu.be/Du_JzbbTuJw
  >- Attempt 2 : With 3 patches in fron of car + **2 orientation with destination** as input    
     >>- https://youtu.be/NW1GH8aQFas

  Sensory data : Taken 3 patches in front of the car

  >- left 30 degrees ; 40x40 
  >- straight 0 degrees ; 40x40
  >- right 30 degrees; 40x40

  > then merged them to create 3 channel input to the CNN Network


  ### Parameters used
  >- **Action Dimension** : 1 - angle of rotation
  >- **State Dimension** : 1200
  >>- 3 channels of 40x40 = 1200 fed to the CNN

  >- Episode **done** if
  >>- Max epsidoes steps reached : 1200
  >>- Car gets to the edges
  >>- Reaches the destination
  
  >- **Rewards**
  >>- On sand = - 1
  >>- Car hits the edges = - 1
  >>- If on Road and distance is reducing = + 1
  >>- On Road = + 0.8

  ### Network

  Convblk1 -> MaxPool -> ConvBlk -> GAP -> FC 

  ### Observations

  Car started to rotate after few episodes.Below is the diagnostics and steps taken. This could mean that only extreme angle (-max_action or + max_action) was predicted. This meant network was unstable. I could not be sure if this was for exploding gradients or vanishing gradients. Took the below approach



  >- Enabled training logs and looked at buffer values ; predicted rotations
  >- Compared training logs between run using T3D + FC + Car(successful) vs T3D + CNN + Car(un-successful)
  
  > Log : https://github.com/senthilva/RL_car_using_T3D/blob/master/logs/analysis_training_logs

          Target Q's should be negative ( they are low positive in CNN network)
            Current Q's should be negative ( they are low positive in CNN network)
            Critic loss should be low positive (they are high positive in CNN network)
            Action Loss should be large positive ( they are los positive in CNN network)
            Q1 should be large negative ( they are small negative in CNN network)

  >- Led me to realize i had not used Batch Normalization across layers. Enabled that and network slightly improved.But, still not going on the road.



## Potential Next Steps

>- Reduce the number of layers in CNN - to see if vanishing gradient in an issue
>- Use only one channel 
>- Get the orientation from destination and feed that to the network and concatenate after feature extraction from CNN
