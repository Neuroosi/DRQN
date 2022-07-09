import torch
from torch import nn
from torch._C import device
from torch import optim
import numpy as np
from collections import deque
from wandb import wandb
import gym
import gym_ple
import vizdoomgym
import DRQN
import DQN
import CNN
from process_state import check_if_enemy_in_obs, clip_reward, makeState, getFrame
import hyperparameters
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)



def train(game):
    env = gym.make(game)
    y = CNN.NeuralNetwork_Recurrent(env.action_space.n, None).to(device)
    target_y = CNN.NeuralNetwork_Recurrent(env.action_space.n, None).to(device)
    y_navigator = CNN.NeuralNetwork_Forward(3, None).to(device)
    y_navigator_target = CNN.NeuralNetwork_Forward(3, None).to(device)
    loss_fn = nn.HuberLoss()
    loss_fn_detector = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(y.parameters(), lr = hyperparameters.learning_rate)
    optimizer_navi = optim.Adam(y_navigator.parameters(), lr = hyperparameters.learning_rate)
    actor = DRQN.DRQN(hyperparameters.REPLAY_MEMORY_SIZE, hyperparameters.BATCH_SIZE, hyperparameters.GAMMA, hyperparameters.EPSILON, hyperparameters.EPSILON_MIN, hyperparameters.EPSILON_DECAY)
    navigator = DQN.DQN(hyperparameters.REPLAY_MEMORY_SIZE, hyperparameters.BATCH_SIZE, hyperparameters.GAMMA, hyperparameters.EPSILON, hyperparameters.EPSILON_MIN, hyperparameters.EPSILON_DECAY)
    state = deque(maxlen = 4)
    print(y)
    answer = input("Use a pre-trained model y/n? ")
    if answer == "y":
        actor.loadModel(y,'actor.pth')
        navigator.loadModel(y_navigator, 'navigato.pth')
    frames_seen = 0
    rewards = []
    avgrewards = []
    wandb.init(project="DRQN_" +game, entity="neuroori") 
    games_played = 0
    total_kills = 0
    for episode in range(1,hyperparameters.EPISODES+500000000000):
        loss = loss_navi = accuracy= None
        obs,labels = env.reset()
        obs = getFrame(obs)
        cumureward = 0
        cumureward2 = 0
        kills = 0 ## 5 for breakout, 3 for spaceinvaders, 0 for pong, 3 for robotank :D
        h = torch.zeros([1,1, 512])
        c  = torch.zeros([1,1, 512])
        while True:
            obs_prev = obs
            enemy_in_frame = check_if_enemy_in_obs(labels)
            if enemy_in_frame == 1.0:
                action, h, c = actor.getPrediction(obs /255,y, h, c)
            else:
                action = navigator.getPrediction(obs / 255, y_navigator)
            ##Repeat same action four times for flappybird/doom otherwise set it to one.

            for repeat in range(hyperparameters.FRAME_SKIP):
                obs, reward, reward2, done, info, labels = env.step(action)
                if done:
                    break
            kills = info["frags"]
            deaths = info["deaths"]
            ##uncomment for atari
            #if info["ale.lives"] < lives:
            #    done = True
            #    lives -= 1
            obs = getFrame(obs)
            #env.render()
            #agent.update_replay_memory((obs_prev, action, clip_reward(reward), obs , done))
            actor.update_replay_memory((obs_prev, action, reward, obs , done, enemy_in_frame))
            if action == 3 or action == 4 or action == 5:
                navigator.update_replay_memory((obs_prev, action, reward2, obs , done))
            ##Train the agent
            if len(actor.replay_memory) >= hyperparameters.START_TRAINING_AT_STEP and frames_seen % hyperparameters.TRAINING_FREQUENCY == 0:
                loss, accuracy = actor.train(y, target_y, loss_fn, loss_fn_detector,  optimizer)
                loss_navi = navigator.train(y_navigator, y_navigator_target, loss_fn,  optimizer_navi)
            ##Update target network  
            if len(actor.replay_memory) >= hyperparameters.START_TRAINING_AT_STEP and frames_seen % hyperparameters.TARGET_NET_UPDATE_FREQUENCY == 0:
                target_y.load_state_dict(y.state_dict())
                y_navigator_target.load_state_dict(y_navigator.state_dict())
                print("Target net updated.")
            frames_seen+=1
            cumureward += reward
            cumureward2 += reward2
            avgrewards.append(np.sum(np.array(rewards))/episode)
            if frames_seen % 10000 == 0:
                actor.saveModel(y,'actor.pth')
                navigator.saveModel(y_navigator, 'navigato.pth')
            if done:
                games_played += 1
                total_kills += kills
                break
            
        print("kills",kills,"avg kills", total_kills/games_played,"deaths", deaths,"Score:", cumureward,"score2:",cumureward2," Episode:", episode, " frames_seen:", frames_seen , " ACTOR_Epsilon:", actor.EPSILON, "NAVI_Epsilon", navigator.EPSILON)
        print(loss, loss_navi, accuracy)
        if loss is not None:
            wandb.log({"avg kills": total_kills/games_played,"kills": kills,"Reward per episode":cumureward, "Avg reward":(np.sum(np.array(rewards))/episode), "Loss":loss, "accuracy": accuracy, "navi loss": loss_navi})

if __name__ == "__main__":
    #game = 'BreakoutDeterministic-v4'
    #game = "SpaceInvadersDeterministic-v4"
    #game = "PongDeterministic-v4"
    #game = "RobotankDeterministic-v4"
    #game = "VizdoomDefendCenter-v0"
    #game = "VizdoomPredictPosition-v0"
    #game = "VizdoomHealthGathering-v0"
    game = "VizdoomDeathmatch-v0"
    #game = 'FlappyBird-v0'
    train(game)