import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
from numpy import mean, std


class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False, start_case=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        avg_speed = {'episodic':0, 'global':[]}
        speed_violation = {'episodic':0, 'global':[]}
        social_violation_cnt = {'episodic':0, 'global':[]}
        personal_violation_cnt = {'episodic':0, 'global':[]}
        jerk_cost = {'episodic':0, 'global':[]}
        aggregated_time = {'episodic':0, 'global':[]}
        side_preference = {'episodic':None, 'global':[]}

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            # If a starting case number is given, reset the environment with case = (start_case + i) every time
            # Otherwise the env.reset() function will increament the case automatically by 1 every time
            if start_case:
                ob = self.env.reset(phase, start_case+i)
            else:
                ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            
            avg_speed['episodic'] = 0
            speed_violation['episodic'] = 0
            social_violation_cnt['episodic'] = 0
            personal_violation_cnt['episodic'] = 0
            jerk_cost['episodic'] = 0
            aggregated_time['episodic'] = 0
            side_preference['episodic'] = None

            while not done:
                action = self.robot.act(ob)
                ob, reward, done, step_info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                # Episodic info logging
                if isinstance(step_info['event'], Discomfort):
                    discomfort += 1
                    min_dist.append(step_info['event'].min_dist)
                avg_speed['episodic'] = avg_speed['episodic'] + (step_info['speed'] - avg_speed['episodic']) / len(states)
                speed_violation['episodic'] = speed_violation['episodic'] + (step_info['speed'] > 1)
                social_violation_cnt['episodic'] += step_info['social_violation_cnt']
                personal_violation_cnt['episodic'] += step_info['personal_violation_cnt']
                jerk_cost['episodic'] += step_info['jerk_cost']
                aggregated_time['episodic'] += step_info['aggregated_time']
                if side_preference['episodic'] is None:
                    side_preference['episodic'] = step_info['side_preference']
                elif step_info['side_preference'] is not None and side_preference['episodic'] != step_info['side_preference']:
                    raise Exception('Side preference changed mid-episode')

            if isinstance(step_info['event'], ReachGoal):
                success += 1
                success_times.append(self.env.global_time)

                # Update logged info to global if episode succeeds
                avg_speed['global'].append(avg_speed['episodic'])
                speed_violation['global'].append(speed_violation['episodic'] / self.env.global_time)
                social_violation_cnt['global'].append(social_violation_cnt['episodic'] / self.env.global_time)
                personal_violation_cnt['global'].append(personal_violation_cnt['episodic'] / self.env.global_time)
                jerk_cost['global'].append(jerk_cost['episodic'] / self.env.global_time)
                aggregated_time['global'].append(aggregated_time['episodic'])
                side_preference['global'].append(side_preference['episodic'])
            elif isinstance(step_info['event'], Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(step_info['event'], Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(step_info['event'], ReachGoal) or isinstance(step_info['event'], Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(mean(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        left_percentage = side_preference['global'].count(0) / len(side_preference['global']) if len(side_preference['global']) > 0 else 0
        right_percentage = side_preference['global'].count(1) / len(side_preference['global']) if len(side_preference['global']) > 0 else 0


        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info(
            f'{phase.upper():<5} {extra_info}has '
            f'success rate: {success_rate:.2f}, '
            f'collision rate: {collision_rate:.2f}, '
            f'nav time: {avg_nav_time:.2f}, '
            f'total reward: {mean(cumulative_rewards):.4f}, '
            f'average return: {mean(average_returns):.4f}, '
            f'social violation: {mean(social_violation_cnt["global"]):.2f}+-{std(social_violation_cnt["global"]):.2f}, '
            f'personal violation: {mean(personal_violation_cnt["global"]):.2f}+-{std(personal_violation_cnt["global"]):.2f}, '
            f'jerk cost: {mean(jerk_cost["global"]):.2f}+-{std(jerk_cost["global"]):.2f}, '
            f'aggregated time: {mean(aggregated_time["global"]):.2f}+-{std(aggregated_time["global"]):.2f}, '
            f'speed: {mean(avg_speed["global"]):.2f}+-{std(avg_speed["global"]):.2f}, '
            f'speed violation: {mean(speed_violation["global"]):.2f}+-{std(speed_violation["global"]):.2f}, '
            f'left %: {left_percentage:.2f}, '
            f'right %: {right_percentage:.2f}')
            

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, mean(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = {
            'success rate': success_rate,
            'collision rate': collision_rate,
            'avg nav time': avg_nav_time,
            'avg cumulative rewards': mean(cumulative_rewards),
            'avg returns': mean(average_returns),
            'avg speed': avg_speed['global'],
            'speed violation': speed_violation['global'],
            'social violation': social_violation_cnt['global'],
            'personal violation': personal_violation_cnt['global'],
            'jerk cost': jerk_cost['global'],
            'aggregated time': aggregated_time['global'],
            'side preference': side_preference['global']}

        return self.statistics

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        self.writer.add_scalar(tag_prefix + '/success_rate', self.statistics['success rate'], global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', self.statistics['collision rate'], global_step)
        self.writer.add_scalar(tag_prefix + '/time', self.statistics['avg nav time'], global_step)
        self.writer.add_scalar(tag_prefix + '/reward', self.statistics['avg cumulative rewards'], global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', self.statistics['avg returns'], global_step)

