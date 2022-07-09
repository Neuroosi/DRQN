import skimage
import numpy as np
import hyperparameters

def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

def getFrame(x):
    #x = x[35:210,0:160]## For breakout, pong
    #x = x[0:405,0:288] #flappybird
    #x = x[75:170,10:160]#For robotank :D
    #x = x[100:480,0:640] ## health gathering
    #x = x[190:480,0:800]
    #x = x[210:405,0:640] ## for predict position
    #x = x[0:200]
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, hyperparameters.INPUTSIZE)
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state = state.astype('uint8')
    #print(state[np.abs(state) > 0])
    #state = np.swapaxes(state, 0, 2 )
    #state = np.swapaxes(state, 1,2 )
    return state

def makeState(state):
    state = np.stack((state[0],state[1],state[2],state[3]), axis=0)
    #state = np.swapaxes(state, 0,1 )
    return state

def check_if_enemy_in_obs(labels):
    if labels is None:
        return 0.0
    for label in labels:
        name = label.object_name
        value = label.value
        if value != 255 and name == 'DoomPlayer':
            return 1.0
    return 0.0