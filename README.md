# RL_car_using_T3D

## Phase 1 
In this project i have taken car environment and integrated it with the T3D learning. I have NOT used CNN in this phase; passed the environment values as scalars into the T3D algorithm. Idea was to first integrate T3D with car environment in Kivy.

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
Started driving on the road after 50 episodes

https://youtu.be/6NWjU9S8zro
