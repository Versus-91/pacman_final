import os
import time
import numpy as np
import torch.optim as optim
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from cnn import PacmanNet
from game import GameWrapper
import random
import matplotlib
import torch
import torch.nn as nn

from run import GameState

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 64
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
MOMENTUM = 0.95
sync_every = 1000
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 1.0
EPS_END = 0.1
MAX_EPISODES = 1500


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.exps = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.exps.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


class PacmanNet(nn.Module):
    def __init__(self):
        super(PacmanNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc_v = nn.Linear(32 * 31 * 28, 256)
        self.relu_v = nn.ReLU()
        self.fc_a = nn.Linear(32 * 31 * 28, 256)
        self.relu_a = nn.ReLU()
        self.fc_value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)

        v = self.fc_v(x)
        v = self.relu_v(v)
        v = self.fc_value(v)

        a = self.fc_a(x)
        a = self.relu_a(a)
        a = self.fc_advantage(a)

        q = v + (a - torch.mean(a, dim=1, keepdim=True))

        return q


class PacmanAgent:
    def __init__(self):
        self.steps = 0
        self.score = 0
        self.target = PacmanNet().to(device)
        self.policy = PacmanNet().to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.memory = ExperienceReplay(20000)
        self.game = GameWrapper()
        self.lr = 0.0003
        self.last_action = 0
        self.buffer = deque(maxlen=4)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.score = 0
        self.episode = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        self.losses = []
        self.epsilon = 1
        self.last_state = GameState()

    def calculate_reward(
        self, done, lives, hit_ghost, action, prev_score, info: GameState, state
    ):
        reward = 0
        time_penalty = -0.01
        movement_penalty = -0.1
        invalid_mode = self.check_cells(info, action)
        progress = round((info.collected_pellets / info.total_pellets) * 10)
        if done:
            if lives > 0:
                print("won")
                reward = 50
            else:
                reward = -50
            return reward
        if self.score - prev_score == 10:
            reward += 1
        if self.score - prev_score == 50:
            print("power up")
            reward += 5
        if reward > 0:
            reward += progress
        if self.score - prev_score >= 200:
            reward += 20 * ((self.score - prev_score) / 200)
        if info.invalid_move and invalid_mode:
            reward -= 10
        if hit_ghost:
            reward -= 30
        reward += time_penalty
        reward += movement_penalty

        index = np.where(info.fram == 5)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            try:
                n1 = info.fram[x + 1][y]
                n2 = info.fram[x - 1][y]
                n3 = info.fram[x][y + 1]
                n4 = info.fram[x][y - 1]
            except IndexError:
                n1 = 0
                n2 = 0
                n3 = 0
                n4 = 0
                print("x", index[0][0], "y", index[1][0])
                print("x", index[0][0], "y", index[1][0])
            if -6 in (n1, n2, n3, n4):
                reward -= 30
            elif 3 in (n1, n2, n3, n4):
                reward += 1 + progress
            elif 4 in (n1, n2, n3, n4):
                reward += 3 + progress
        if action == REVERSED[self.last_action]:
            reward -= 5
        reward = round(reward, 2)
        return reward

    def get_reward(self, done, lives, hit_ghost, action, prev_score, info: GameState):
        reward = 0
        invalid_move = self.check_cells(info, action)
        if done:
            if lives > 0:
                print("won")
                reward = 10
            else:
                reward = -10
            return reward
        progress = int((info.collected_pellets / info.total_pellets) * 5)
        if self.score - prev_score == 10:
            reward += 4
        if self.score - prev_score == 50:
            reward += 5
        if self.score - prev_score >= 200:
            reward += 3
        if hit_ghost:
            reward -= 10
        if info.invalid_move and invalid_move:
            reward -= 1
        if info.invalid_move and self.last_state.stopped:
            reward -= 1
        reward -= 1
        return reward

    def write_matrix(self, matrix):
        with open("outfile.txt", "wb") as f:
            for line in matrix:
                np.savetxt(f, line, fmt="%.2f")

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
        predicted_Q_values = self.policy(state_batch)
        predicted_state_action_values = predicted_Q_values.gather(1, action_batch)
        with torch.no_grad():
            next_Q_values_target = self.target(new_state_batch)
            next_state_values = next_Q_values_target.max(1)[0]

        target_values = reward_batch + GAMMA * (1 - dones) * next_state_values
        target_values = target_values.unsqueeze(1)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_state_action_values, target_values).to(device)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def act(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        rand = random.random()
        self.steps += 1
        if rand > self.epsilon:
            with torch.no_grad():
                outputs = self.policy(state)
            return outputs.max(1)[1].view(1, 1)
        else:
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self, items, name="plot.png", label="rewards", avg=100):
        plt.figure(1)
        rewards = torch.tensor(items, dtype=torch.float)
        plt.xlabel("Episode")
        plt.ylabel(label)
        plt.plot(rewards.numpy())
        if len(rewards) >= avg:
            means = rewards.unfold(0, avg, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg - 1), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        plt.savefig(name)

    def process_state(self, states):
        tensor = [torch.from_numpy(arr).float().to(device) for arr in states]

        # frightened_ghosts_tensor = torch.from_numpy(
        #     states[3]).float().to(device)
        channel_matrix = torch.stack(tensor, dim=0)
        channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self, force=False):
        if (self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0) or force:
            torch.save(
                self.policy.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"policy-model-{self.episode}-{self.steps}.pt",
                ),
            )
            torch.save(
                self.target.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"target-model-{self.episode}-{self.steps}.pt",
                ),
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"optimizer-{self.episode}-{self.steps}.pt",
                ),
            )

    def load_model(self, name, eval=False):
        path = os.path.join(os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))

        if eval:
            self.target.eval()
            self.policy.eval()
        else:
            path = os.path.join(os.getcwd() + "\\results", f"optimizer-{name}.pt")
            self.optimizer.load_state_dict(torch.load(path))
            name_parts = name.split("-")
            self.episode = int(name_parts[0])
            self.steps = int(name_parts[1])
            self.target.train()
            self.policy.train()

    def check_cells(self, info, action):
        row_indices, col_indices = np.where(info.frame == 5)
        invalid_in_maze = False
        if row_indices.size > 0:
            x = row_indices[0]
            y = col_indices[0]
            try:
                upper_cell = info.frame[x - 1][y]
                lower_cell = info.frame[x + 1][y]
                right_cell = info.frame[x][y + 1]
                left_cell = info.frame[x][y - 1]
            except IndexError:
                upper_cell = 0
                lower_cell = 0
                right_cell = 0
                left_cell = 0
            if info.invalid_move:
                if action == 0:
                    if upper_cell == 1:
                        invalid_in_maze = True
                elif action == 1:
                    if lower_cell == 1:
                        invalid_in_maze = True
                elif action == 2:
                    if left_cell == 1:
                        invalid_in_maze = True
                elif action == 3:
                    if right_cell == 1:
                        invalid_in_maze = True
        return invalid_in_maze

    def train(self):
        if self.episode >= MAX_EPISODES:
            self.save_model(force=True)
            exit()
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        obs, self.score, done, info = self.game.step(random_action)
        # state = self.process_state(obs)
        tensor = [torch.from_numpy(arr).float().to(device) for arr in info.state]
        state = torch.stack(tensor, dim=0).unsqueeze(0)
        # for i in range(6):
        #     obs, self.score, done, info = self.game.step(random_action)
        #     self.buffer.append(info.frame)
        # state = self.process_state(self.buffer)
        last_score = 0
        lives = 3
        reward_total = 0
        while True:
            prev = info.state
            action = self.act(state)
            action_t = action.item()
            for i in range(3):
                if not done:
                    obs, self.score, done, info = self.game.step(action_t)
                    if lives != info.lives:
                        break

            # self.buffer.append(info.frame)
            hit_ghost = False
            if lives != info.lives:
                # self.write_matrix(self.buffer)
                hit_ghost = True
                lives -= 1
                if not done:
                    for i in range(3):
                        _, _, _, _ = self.game.step(action_t)
            tensor = [torch.from_numpy(arr).float().to(device) for arr in info.state]
            next_state = torch.stack(tensor, dim=0).unsqueeze(0)
            # next_state = self.process_state(self.buffer)

            reward_ = self.get_reward(
                done, lives, hit_ghost, action_t, last_score, info
            )
            self.last_state = info
            reward_total += reward_
            last_score = self.score
            action_tensor = torch.tensor([[action_t]], device=device, dtype=torch.long)
            self.memory.append(
                state,
                action_tensor,
                torch.tensor([reward_], device=device),
                next_state,
                done,
            )
            state = next_state
            # if self.steps % 2 == 0:
            self.learn()
            if not info.invalid_move:
                self.last_action = action_t
            # if self.steps % 100000 == 0:
            #     self.scheduler.step()
            if done:
                self.epsilon = max(
                    EPS_END,
                    EPS_START - (EPS_START - EPS_END) * (self.episode) / MAX_EPISODES,
                )
                self.log()
                self.rewards.append(self.score)
                self.plot_rewards(items=self.rewards, avg=50)
                # self.plot_rewards(items = self.losses,label="losses",name="losses.png", avg=50)
                time.sleep(1)
                self.game.restart()
                torch.cuda.empty_cache()
                break

    def log(self):
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            "epsilon",
            round(self.epsilon, 3),
            "reward",
            self.score,
            "learning rate",
            current_lr,
            "episode",
            self.episode,
            "steps",
            self.steps,
        )

    def test(self, episodes=10):
        if self.episode < episodes:
            obs = self.game.start()
            self.episode += 1
            random_action = random.choice([0, 1, 2, 3])
            for i in range(6):
                obs, self.score, done, info = self.game.step(random_action)
                self.buffer.append(info.frame)
            state = self.process_state(self.buffer)
            lives = 3
            while True:
                action = self.act(state, eval=True)
                action_t = action.item()
                print(action_t)
                for i in range(3):
                    if not done:
                        obs, reward, done, info = self.game.step(action_t)
                    if lives != info.lives:
                        lives = info.lives
                        break
                self.buffer.append(info.frame)
                state = self.process_state(self.buffer)
                if done:
                    self.rewards.append(reward)
                    self.plot_rewards(name="test.png", items=self.rewards, avg=2)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            exit()


if __name__ == "__main__":
    agent = PacmanAgent()
    # agent.load_model(name="300-79595", eval=False)
    while True:
        agent.train()
        # agent.test()
