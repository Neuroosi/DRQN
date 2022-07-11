import gym
from gym import spaces
import vizdoom.vizdoom as vzd
from vizdoom.vizdoom import GameVariable, Button
import numpy as np
import os
from typing import List

turn_off_rendering = False
try:
    from gym.envs.classic_control import rendering
except Exception as e:
    print(e)
    turn_off_rendering = True
    


# Rewards
# 1 per kill
reward_factor_frag = 1.0
reward_factor_damage = 0.01

# Player can move at ~16.66 units per tick
reward_factor_distance = 5e-4
penalty_factor_distance = -2.5e-3
reward_threshold_distance = 3.0

# Pistol clips have 10 bullets
reward_factor_ammo_increment = 0.002
reward_factor_ammo_decrement = -0.001

# Player starts at 100 health
reward_factor_health_increment = 0.02
reward_factor_health_decrement = -0.01
reward_factor_armor_increment = 0.01

#MAPS = ["map01", "map02", "map03", "map04", "map05", "map06", "map07", "map08", "map09", "map10", "map11",  "map13", "map14", "map15", "map16", "map17", "map18", "map19", "map20"]
#MAPS = ["map33"]
CONFIGS = [
    ["basic.cfg", 3],  # 0
    ["deadly_corridor.cfg", 7],  # 1
    ["defend_the_center.cfg", 3],  # 2
    ["defend_the_line.cfg", 3],  # 3
    ["health_gathering.cfg", 3],  # 4
    ["my_way_home.cfg", 5],  # 5
    ["predict_position.cfg", 3],  # 6
    ["take_cover.cfg", 2],  # 7
    ["deathmatch.cfg", 6],  # 8
    ["health_gathering_supreme.cfg", 3],  # 9
]

# List of game variables storing ammunition information. Used for keeping track of ammunition-related rewards.
AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
                  GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]

# List of game variables storing weapon information. Used for keeping track of ammunition-related rewards.
WEAPON_VARIABLES = [GameVariable.WEAPON0, GameVariable.WEAPON1, GameVariable.WEAPON2, GameVariable.WEAPON3,
                    GameVariable.WEAPON4,
                    GameVariable.WEAPON5, GameVariable.WEAPON6, GameVariable.WEAPON7, GameVariable.WEAPON8,
                    GameVariable.WEAPON9]

class VizdoomEnv(gym.Env):
    def __init__(self, level, **kwargs):
        """
        Base class for Gym interface for ViZDoom. Child classes are defined in vizdoom_env_definitions.py,
        that contain the level parameter and pass through any kwargs from gym.make()
        :param level: index of level in the CONFIGS list above
        :param kwargs: keyword arguments from gym.make(env_name_string, **kwargs) call. 'depth' will render the
        depth buffer and 'labels' will render the object labels and return it in the observation.
        Note that the observation will be a list with the screen buffer as the first element. If no kwargs are
        provided (or depth=False and labels=False) the observation will be of type np.ndarray.
        """

        # init game
        self.game = vzd.DoomGame()
        self.game.set_labels_buffer_enabled(True)
        self.game.set_doom_map("map07")
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[level][0]))
        self.game.set_window_visible(False)
        self.game.clear_available_game_variables()
        self.game.add_game_args('-host 1 -deathmatch 1 -nomonsters 1')  
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.ARMOR)
        for i in range(len(AMMO_VARIABLES)):
            self.game.add_available_game_variable(AMMO_VARIABLES[i])
        for i in range(len(WEAPON_VARIABLES)):
            self.game.add_available_game_variable(WEAPON_VARIABLES[i])
        self.game.init()
        for i in range(8):
            self.game.send_game_command('addbot')
      
        self.state = None
        self.viewer = None
        self.last_damage_dealt = 0
        self.last_health = 0
        self.last_armor = 0
        self.ammo_state = 0
        self.last_x = 0
        self.last_y = 0
        self.last_frags = 0
        self.deaths = 0
        self.action_space = spaces.Discrete(CONFIGS[level][1])
        self.rewards_stats = {
            'frag': 0,
            'damage': 0,
            'ammo': 0,
            'health': 0,
            'armor': 0,
            'distance': 0,
        }
        # specify observation space(s)
        list_spaces: List[gym.Space] = [
            spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.game.get_screen_channels(),
                ),
                dtype=np.uint8,
            )
        ]

        if len(list_spaces) == 1:
            self.observation_space = list_spaces[0]
        else:
            self.observation_space = spaces.Tuple(list_spaces)

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()
        
        reward = self.game.make_action(act, 4)
        #self.last_frags += np.sign(reward)
        self._respawn_if_dead()
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()
        info = {"frags": self.last_frags, "deaths": self.deaths, "deaths2": self.game.get_game_variable(GameVariable.DEATHCOUNT)}
        reward, reward2 = self.shape_rewards()
        if done is False:
        	return self.__collect_observations(), reward, reward2, done, info, self.state.labels
        return self.__collect_observations(), reward, reward2, done, info, None
        
    def shape_rewards(self):
        reward_contributions = [
            self._compute_frag_reward(),
            self._compute_damage_reward(),
            self._compute_ammo_reward(),
            self._compute_health_reward(),
            self._compute_armor_reward(),
            self._compute_distance_reward(*self._get_player_pos()),
        ]
        
        
        reward = np.sum(np.array(reward_contributions[:-1]))
        if reward_contributions[2] < 0:
             reward_contributions[2] = 0
        if reward_contributions[4] < 0:
             reward_contributions[4] = 0
        if reward_contributions[3] < 0:
             reward_contributions[3] = 0
        if reward_contributions[3] < 0:
             reward_contributions[3] = 0
        return reward, np.sum(np.array(reward_contributions[2:]))
        
    def _compute_frag_reward(self):
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = reward_factor_frag * (frags - self.last_frags)
        self.last_frags = frags
        self._log_reward_stat('frag', reward)
        return reward
    
    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            # Check if player is dead
            if self.game.is_player_dead():
                self.deaths += 1
                self._reset_player()

    def _compute_distance_reward(self, x, y):
        """Computes a reward/penalty based on the distance travelled since last update."""
        dx = self.last_x - x
        dy = self.last_y - y

        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance - reward_threshold_distance > 0:
            reward = distance * reward_factor_distance
        else:
            reward = -reward_factor_distance

        self.last_x = x
        self.last_y = y
        self._log_reward_stat('distance', reward)

        return reward

    def _compute_damage_reward(self):
        """Computes a reward based on total damage inflicted to enemies since last update."""
        damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
        reward = reward_factor_damage * (damage_dealt - self.last_damage_dealt)
        self.last_damage_dealt = damage_dealt
        self._log_reward_stat('damage', reward)

        return reward

    def _compute_health_reward(self):
        """Computes a reward/penalty based on total health change since last update."""
        # When the player is dead, the health game variable can be -999900
        health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)

        health_reward = reward_factor_health_increment * max(0, health - self.last_health)
        health_penalty = reward_factor_health_decrement * min(0, health - self.last_health)
        reward = health_reward - health_penalty

        self.last_health = health
        self._log_reward_stat('health', reward)

        return reward

    def _compute_armor_reward(self):
        """Computes a reward/penalty based on total armor change since last update."""
        armor = self.game.get_game_variable(GameVariable.ARMOR)
        reward = reward_factor_armor_increment * max(0, armor - self.last_armor)
        
        self.last_armor = armor
        self._log_reward_stat('armor', reward)

        return reward

    def _compute_ammo_reward(self):
        """Computes a reward/penalty based on total ammunition change since last update."""
        self.weapon_state = self._get_weapon_state()

        new_ammo_state = self._get_ammo_state()
        ammo_diffs = (new_ammo_state - self.ammo_state) * self.weapon_state
        ammo_reward = reward_factor_ammo_increment * max(0, np.sum(ammo_diffs))
        ammo_penalty = reward_factor_ammo_decrement * min(0, np.sum(ammo_diffs))
        reward = ammo_reward - ammo_penalty
        
        self.ammo_state = new_ammo_state
        self._log_reward_stat('ammo', reward)

        return reward

    def _get_player_pos(self):
        """Returns the player X- and Y- coordinates."""
        return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
            GameVariable.POSITION_Y)

    def _get_ammo_state(self):
        """Returns the total available ammunition per weapon slot."""
        ammo_state = np.zeros(10)

        for i in range(10):
            ammo_state[i] = self.game.get_game_variable(AMMO_VARIABLES[i])

        return ammo_state

    def _get_weapon_state(self):
        """Returns which weapon slots can be used. Available weapons are encoded as ones."""
        weapon_state = np.zeros(10)

        for i in range(10):
            weapon_state[i] = self.game.get_game_variable(WEAPON_VARIABLES[i])

        return weapon_state

    def _log_reward_stat(self, kind: str, reward: float):
        self.rewards_stats[kind] += reward

    def _reset_player(self):
        self.last_health = 100
        self.last_armor = 0
        self.game.respawn_player()
        self.last_x, self.last_y = self._get_player_pos()
        self.ammo_state = self._get_ammo_state()
        
    def _reset_bots(self):
    # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command('removebots')
        for i in range(8):
            self.game.send_game_command('addbot')

    def reset(self):
        #mapp = np.random.choice(MAPS)
        #print("map", mapp)
        #self.game.set_doom_map(mapp)
        self.game.new_episode()
        self.state = self.game.get_state()
        self.last_health = 100
        self.last_x, self.last_y = self._get_player_pos()
        self.last_armor = self.last_frags  = self.deaths = 0
        self._reset_bots()
        for k in self.rewards_stats.keys():
            self.rewards_stats[k] = 0
        return self.__collect_observations(),self.state.labels
        

    def __collect_observations(self):
        observation = []
        if self.state is not None:
            observation.append(np.transpose(self.state.screen_buffer, (1, 2, 0)))
        else:
            # there is no state in the terminal step, so a "zero observation is returned instead"
            if isinstance(self.observation_space, gym.spaces.box.Box):
                # Box isn't iterable
                obs_space = [self.observation_space]
            else:
                obs_space = self.observation_space

            for space in obs_space:
                observation.append(np.zeros(space.shape, dtype=space.dtype))

        # if there is only one observation, return obs as array to sustain compatibility
        if len(observation) == 1:
            observation = observation[0]
        return observation

    def render(self, mode="human"):
        if turn_off_rendering:
            return
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        except AttributeError:
            pass
        
    def close(self):
        if self.viewer:
            self.viewer.close()

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {
            (): 2,
            (ord("a"),): 0,
            (ord("d"),): 1,
            (ord("w"),): 3,
            (ord("s"),): 4,
            (ord("q"),): 5,
            (ord("e"),): 6,
        }
        return keys
