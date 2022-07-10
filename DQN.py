import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DQN(nn.Module):

    def __init__(self, replay_memory_size = 2*10**5, batch_size = 32, gamma = 0.99, epsilon = 1, epsilon_min = 0.1,  epsilon_decay = 250000):
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.ddqn = False
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPSILON = 1
        self.EPSILON_MIN = epsilon_min
        self.EPSILON_DECAY = (self.EPSILON-self.EPSILON_MIN)/epsilon_decay

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, agent, target, loss_fn, optimizer):
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        Y = []
        states = [torch.from_numpy(np.array(transition[0])/255) for transition in batch]
        states = torch.stack(states)
        states = states.float()
        #states = torch.unsqueeze(states,axis = 1)
        next_states = [torch.from_numpy(np.array(transition[3])/255) for transition in batch]
        next_states = torch.stack(next_states)
        next_states = next_states.float()
        #next_states = torch.unsqueeze(next_states,axis = 1)
        optimizer.zero_grad()
        y = agent(states)
        target_y = target(next_states)
        y_next = agent(next_states)
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if action == 3:
                action =  0
            elif action == 4:
                action = 1
            else:
                action = 2
            if done:
                y[i][action] = reward
            elif self.ddqn is False:
                y[i][action] = reward + self.GAMMA*torch.max(target_y[i])
            else:
                y[i][action] = reward + self.GAMMA*target_y[i][torch.argmax(y_next[i])]
            Y.append(y[i])
        Y = torch.stack(Y)
        agent.train()
        pred = agent(states)
        loss = loss_fn(pred, Y)
        loss.backward()
        for param in agent.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        self.EPSILON = max(self.EPSILON_MIN, self.EPSILON-self.EPSILON_DECAY)
        return loss.item()


    def getPrediction(self, state, model):
        if np.random.rand() > self.EPSILON:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.from_numpy(state)
                state = state.float()
                state = torch.unsqueeze(state,axis = 0)
                output = model(state)
                action = torch.argmax(output).item()
                if action == 0:
                    return 3
                elif action == 1:
                    return 4
                else:
                    return 5
        action = random.randrange(model.actionSpaceSize)
        if action == 0:
            return 3
        elif action == 1:
            return 4
        else:
            return 5

    def saveModel(self, agent, filename):
        torch.save(agent.state_dict(), filename)
        print("Model saved!")
    def loadModel(self, agent, filename):
        agent.load_state_dict(torch.load(filename))
        print("Model loaded!")