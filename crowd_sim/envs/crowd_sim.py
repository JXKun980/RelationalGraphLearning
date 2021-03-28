import logging
import random
import math

import gym
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.shape import AgentHeadingRect


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None
        self.all_scenarios = ['circle_crossing', 'square_crossing', 'parallel_traffic', 'perpendicular_traffic', '2_agents_passing', '2_agents_overtaking', '2_agents_crossing']
        self.all_multi_agent_scenarios = ['circle_crossing', 'square_crossing', 'parallel_traffic', 'perpendicular_traffic']
        self.scenario_cnt = 0

        # Override default human safety distance if set
        self.human_safety_space = None

        # Evaluation
        self.infos = None
        self.last_acceleration = (0,0)
        self.min_personal_space = 0.2

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def set_human_safety_space(self, safety_space):
        self.human_safety_space = safety_space
        if self.centralized_planning:
            self.centralized_planner.safety_space = safety_space

    def generate_human(self, human=None):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if self.human_safety_space:
            human.policy.safety_space = self.human_safety_space
        
        return human

    def generate_scenario(self, scenario, human_num, case_counter):
        if scenario == 'randomized_multiagent_scenarios':
            scenario = random.choice(self.all_multi_agent_scenarios)
        elif scenario == 'all_multiagent_scenarios':
            case_counter = case_counter % len(self.all_multi_agent_scenarios)
            scenario = self.all_multi_agent_scenarios[case_counter]
    
        if scenario == 'circle_crossing':
            # Robot
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # Human
            self.humans = []
            for _ in range(human_num):
                human = self.generate_human()
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    safe = self.is_safe_agent_spawn(px, py, human.radius) and self.is_safe_goal_spawn(-px, -py, human.radius)
                    if safe:
                        break
                human.set(px, py, -px, -py, 0, 0, 0)
                self.humans.append(human)
                
        elif scenario == 'square_crossing':
            # Robot
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # Human
            self.humans = []
            for _ in range(human_num):
                human = self.generate_human()
                while True:
                    px = (np.random.random() - 0.5) * self.square_width * 0.4
                    py = (np.random.random() - 0.5) * self.square_width * 0.4
                    safe = self.is_safe_agent_spawn(px, py, human.radius)
                    if safe:
                        break
                while True:
                    gx = (np.random.random() - 0.5) * self.square_width * 0.4
                    gy = (np.random.random() - 0.5) * self.square_width * 0.4
                    safe = self.is_safe_goal_spawn(gx, gy, human.radius)
                    if safe:
                        break
                human.set(px, py, gx, gy, 0, 0, 0)
                self.humans.append(human)

        elif scenario == '2_agents_passing':
            # Robot
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # Human
            self.human_num = 1
            human = self.generate_human()

            min_x = - (self.robot.radius + human.radius)
            max_x = self.robot.radius + human.radius
            max_num_cases = 10

            human_x = case_counter % max_num_cases * (max_x - min_x) / max_num_cases + min_x

            human.set(human_x, self.circle_radius, human_x, -self.circle_radius, 0, 0, 0)
            self.humans = [human]

        elif scenario == '2_agents_overtaking':
            # Robot
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # Human
            self.human_num = 1
            human = self.generate_human()
            human.v_pref = 0.3

            min_x = - (self.robot.radius + human.radius)
            max_x = self.robot.radius + human.radius
            max_num_cases = 10

            human_x = case_counter % max_num_cases * (max_x - min_x) / max_num_cases + min_x

            human.set(human_x, -self.circle_radius+2, human_x, self.circle_radius+2, 0, 0, 0)
            self.humans = [human]

        elif scenario == '2_agents_crossing':
            # Robot
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # Human
            self.human_num = 1
            human = self.generate_human()

            min_x = -(self.circle_radius + self.robot.radius + human.radius)
            max_x = -(self.circle_radius - self.robot.radius - human.radius)
            max_num_cases = 10

            human_x = case_counter % max_num_cases * (max_x - min_x) / max_num_cases + min_x

            human.set(human_x, 0, -human_x, 0, 0, 0, 0)
            self.humans = [human]

        elif scenario == 'parallel_traffic':
            # Robot
            self.robot.set(0, -4, 0, 4, 0, 0, np.pi / 2)
            # Human
            self.humans = []
            for _ in range(human_num):
                human = self.generate_human()
                while True:
                    px = (np.random.random() - 0.5) * (0.4 * self.panel_width)
                    if np.random.random() >= 0.5: # Sign determine if the traffic is oppsite or on the same direction
                        sign = 1
                    else:
                        sign = -1
                    py = np.random.random() * (0.4 * self.panel_height) * sign
                    gx = px
                    gy = np.random.random() * (0.4 * self.panel_height) * -sign
                    safe = self.is_safe_agent_spawn(px, py, human.radius) and self.is_safe_goal_spawn(gx, gy, human.radius)
                    if safe:
                        break
                human.set(px, py, gx, gy, 0, 0, 0)
                self.humans.append(human)

        elif scenario == 'perpendicular_traffic':
            # Robot
            self.robot.set(0, -4, 0, 4, 0, 0, np.pi / 2)
            # Human
            self.humans = []
            for _ in range(human_num):
                human = self.generate_human()
                while True:
                    px = np.random.random() * (-0.5 * self.panel_width)
                    py = np.random.random() * (0.9 * self.panel_height) - (0.5 * self.panel_height)
                    gx = self.panel_width
                    gy = py
                    safe = self.is_safe_agent_spawn(px, py, human.radius) and self.is_safe_goal_spawn(gx, gy, human.radius)
                    if safe:
                        break
                human.set(px, py, gx, gy, 0, 0, 0)
                self.humans.append(human)

        elif scenario == 'group cutting':
            raise NotImplementedError()


        else:
            raise Exception('Unknown scenario passed in')
    
    def is_safe_agent_spawn(self, px, py, radius):
        safe = True
        for agent in [self.robot] + self.humans:
            if norm((px - agent.px, py - agent.py)) < radius + agent.radius + self.discomfort_dist:
                safe = False
                break
        return safe

    def is_safe_goal_spawn(self, gx, gy, radius):
        safe = True
        for agent in [self.robot] + self.humans:
            if norm((gx - agent.gx, gy - agent.gy)) < radius + agent.radius + self.discomfort_dist:
                safe = False
                break
        return safe

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}

        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))

            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                self.human_num = 1
                self.test_scenario = 'circle_crossing'

            # Generate scenarios based on simulation config
            self.generate_scenario(self.test_scenario, self.human_num, self.case_counter[phase])
            # print('case_counter: {}'.format(self.case_counter[phase]))
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        self.infos = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        # get human actions
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        # social score violation calculation
        heading_rect_violations = 0
        robot_rect = AgentHeadingRect(self.robot.px, self.robot.py, self.robot.radius, self.robot.vx, self.robot.vy, self.robot.kinematics, action)
        for human in self.humans:
            human_rect = AgentHeadingRect(human.px, human.py, human.radius, human.vx, human.vy, human.kinematics)
            if robot_rect.intersects(human_rect):
                heading_rect_violations += 1
            

        # collision detection
        dmin = float('inf')
        collision = False
        
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        # calculate step jerk cost
        ax = action.vx - self.robot.vx
        ay = action.vy - self.robot.vy
        d_ax = ax - self.last_acceleration[0]
        d_ay = ay - self.last_acceleration[1]
        jerk_cost = d_ax**2 + d_ay**2
        self.last_acceleration = (ax, ay)

        # state information logging
        info = {}
        info['aggregated_time'] = 1
        info['min_separation'] = dmin
        info['social_violation_cnt'] = heading_rect_violations
        info['jerk_cost'] = jerk_cost
        info['speed'] = math.sqrt(action.vx**2 + action.vy**2)

        if dmin < self.min_personal_space:
            info['personal_violation_cnt'] = 1
        else:
            info['personal_violation_cnt'] = 0

        if self.global_time >= self.time_limit - 1:
            info['aggregated_time'] = math.inf
            reward = 0
            done = True
            info['event'] = Timeout()
        elif collision:
            info['aggregated_time'] = math.inf
            reward = self.collision_penalty
            done = True
            info['event'] = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info['event'] = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info['event'] = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info['event'] = Nothing()
        
        for human in self.humans:
            if not human.reached_destination():
                info['aggregated_time'] += 1

        # update environment
        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self.generate_human(human)

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)
            self.infos.append(info)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='video', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = False

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            show_sensor_range = True
            show_eval_info = False
            show_social_zone = False

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            if show_human_start_goal:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8, label='Start')
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            # add robot and its goal 
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal, robot_start], ['Robot', 'Goal', 'Start'], fontsize=14)
            # if show_sensor_range:
            #     sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            #     ax.add_artist(sensor_range)


            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i))
                      for i in range(len(self.humans))]

            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]
            
            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.5, 0.9, f'Time: {0}', fontsize=16, transform=ax.transAxes, horizontalalignment='center',
                verticalalignment='center')
            ax.add_artist(time)

            # add evaluation annotation
            if show_eval_info:
                eval_text = plt.text(0.6, 0.07, 
                    f"Aggregated Time: {0}\nMinimum Separation: {0}\nSocial Zone Violations: {0}\nJerk Cost: {0}",
                    fontsize=12, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')

            # calculate evaluation information
            list_aggregated_time = [self.infos[0]['aggregated_time']]
            list_min_separation = [self.infos[0]['min_separation']]
            list_personal_violation_cnt = [self.infos[0]['personal_violation_cnt']]
            list_social_violation_cnt = [self.infos[0]['social_violation_cnt']]
            list_jerk_cost = [self.infos[0]['jerk_cost']]
            for i in range(1, len(self.infos)):
                list_aggregated_time.append(list_aggregated_time[i-1] + self.infos[i]['aggregated_time'])
                list_min_separation.append(min(list_min_separation[i-1], self.infos[i]['min_separation']))
                list_social_violation_cnt.append(list_social_violation_cnt[i-1] + self.infos[i]['social_violation_cnt'])
                list_jerk_cost.append(list_jerk_cost[i-1] + self.infos[i]['jerk_cost'])
                list_personal_violation_cnt.append(list_personal_violation_cnt[i-1] + self.infos[i]['personal_violation_cnt'])

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute social zone for each step
            social_zones_all_agents = []
            
            if show_social_zone:
                for i in range(self.human_num + 1):
                    social_zones = []
                    step_cnt = 0
                    for state in self.states:
                        step_cnt += 1
                        agent_state = state[0] if i == self.human_num else state[1][i]
                        if i == self.human_num: # robot
                            rect = AgentHeadingRect(agent_state.px, agent_state.py, self.robot.radius, agent_state.vx, agent_state.vy, self.robot.kinematics)
                            if step_cnt < len(self.infos) and self.infos[step_cnt]['social_violation_cnt'] > 0:
                                rect.color = 'red'
                        else:
                            rect = AgentHeadingRect(agent_state.px, agent_state.py, self.humans[i].radius, agent_state.vx, agent_state.vy, self.humans[i].kinematics)
                        social_zones.append(rect.get_pyplot_rect())
                    social_zones_all_agents.append(social_zones)

            # draw the zones for the first step
            social_zones_drawn = []
            for zones in social_zones_all_agents:
                ax.add_artist(zones[0])
                social_zones_drawn.append(zones[0])

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0: # =========================================================== TODO: why unicycle only?
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                           agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)
                if i == 0:
                    arrow_color = 'black'
                    arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])
            for arrow in arrows:
                ax.add_artist(arrow)

            global_step = 0

            if len(self.trajs) != 0:
                human_future_positions = []
                human_future_circles = []
                for traj in self.trajs:
                    human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
                                              for step in range(self.robot.policy.planning_depth)]
                                             for i in range(self.human_num)]
                    human_future_positions.append(human_future_position)

                for i in range(self.human_num):
                    circles = []
                    for j in range(self.robot.policy.planning_depth):
                        circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
                        ax.add_artist(circle)
                        circles.append(circle)
                    human_future_circles.append(circles)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal social_zones_drawn
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    if display_numbers:
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
                for arrow in arrows:
                    arrow.remove()
                for zone in social_zones_drawn: # remove last step's social zones
                    zone.remove()

                # draw social zones for each step
                if show_social_zone:
                    social_zones_drawn = []
                    for i in range(self.human_num + 1):
                        zones = social_zones_all_agents[i]
                        social_zones_drawn.append(zones[frame_num])
                        ax.add_artist(zones[frame_num])

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                if show_eval_info:
                    eval_text.set_text(f"Aggregated Time: {list_aggregated_time[frame_num]}\
                        \nPersonal Zone Violations: {list_personal_violation_cnt[frame_num]}\
                        \nSocial Zone Violations: {list_social_violation_cnt[frame_num]}\
                        \nJerk Cost: {list_jerk_cost[frame_num]: .3f}")

                if len(self.trajs) != 0:
                    for i, circles in enumerate(human_future_circles):
                        for j, circle in enumerate(circles):
                            circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500, repeat_delay=500)
            anim.running = True

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                plt.rcParams["animation.convert_path"] = r'/usr/bin/convert'
                anim.save(output_file, writer='imagemagick', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError
