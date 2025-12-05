import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class HMM_target_detector(object):

    def __init__(self,
                 emission_matrix_targets:np.ndarray, 
                 max_iter = 100,
                 angle_span=2*np.pi,
                 CODE_COLOR_MAP = {0 : 'yellow',
                                1 : 'blue',
                                2 : 'purple'}):

        self.emission_matrix_targets = emission_matrix_targets
        self.num_states = emission_matrix_targets.shape[2]
        self.num_observations = emission_matrix_targets.shape[1]
        self.num_beliefs = emission_matrix_targets.shape[0]

        self.emission_matrix_true = emission_matrix_targets[0]
        self.color_code_map = CODE_COLOR_MAP

        self.state_step = angle_span/self.num_states
        self.candidate_steps = np.array([-self.state_step, 0, self.state_step])
        self.state_centers = (np.arange(-self.state_step, angle_span-self.state_step, self.state_step) + self.state_step) % angle_span

        self.prior_target_prob = np.full(self.num_beliefs, 1/self.num_beliefs)
        self.prior_state_prob = np.full(self.num_states, 1/self.num_states)       

        self.within_trial_params = dict()
        self.within_trial_params['time_ind'] = 0
        self.within_trial_params['initial_angle'] = 0
        self.within_trial_params['max_iter'] = max_iter
        self.within_trial_params['stopping_num'] = 0

        self.within_trial_arrays = dict()
        self.within_trial_arrays['joint_prob_obs_state_t_from_targets'] = np.zeros((max_iter, self.num_beliefs, self.num_observations, self.num_states))
        self.within_trial_arrays['joint_prob_received_obs_state_t_from_targets'] = np.zeros((max_iter, self.num_beliefs, self.num_states))
        self.within_trial_arrays['joint_prob_received_obs_state_forecast_t_fromtargets'] = np.zeros((max_iter, self.num_beliefs, self.num_states))
        self.within_trial_arrays['likelihood_received_obs_from_targets_t'] = np.zeros((max_iter, self.num_beliefs))

        self.within_trial_arrays['angles_visited'] = np.zeros(max_iter)
        self.within_trial_arrays['steps_taken'] = np.zeros(max_iter)
        self.within_trial_arrays['posterior_t'] = np.zeros((max_iter, self.num_beliefs))
        self.within_trial_arrays['current_entropyS_t'] = np.zeros(max_iter)
        
        self.within_trial_arrays['expected_forecast_t_S_per_step'] = np.zeros((max_iter, self.candidate_steps.shape[0]))
        self.within_trial_arrays['deltaS_t'] = np.zeros((max_iter, self.candidate_steps.shape[0]))
        self.within_trial_arrays['code_received_t'] = np.zeros(max_iter)
        self.within_trial_arrays['decision_type_t'] = np.empty(max_iter, dtype=np.dtypes.StringDType())


    def get_deterministic_state_transition_matrix(self, candidate_step):
        if candidate_step!=0:
            shift = int(candidate_step / self.state_step)
        else:
            shift = 0

        state_matrix = np.eye(self.num_states)
        state_transition_matrix = np.roll(state_matrix, shift, axis=1)
        
        return state_transition_matrix

    def move_and_sample_from_object(self):
        if self.within_trial_params["time_ind"]<1:
            self.within_trial_arrays['angles_visited'][self.within_trial_params["time_ind"]] = self.within_trial_params["initial_angle"]
        else:
            self.within_trial_arrays['angles_visited'][self.within_trial_params["time_ind"]] = self.within_trial_arrays['angles_visited'][self.within_trial_params["time_ind"]-1] + np.rad2deg(self.within_trial_arrays['steps_taken'][int(self.within_trial_params["time_ind"]-1)])

        angle_visited_radians = np.radians((self.within_trial_arrays['angles_visited'][self.within_trial_params["time_ind"]] % 360))
        state_visited = np.floor((angle_visited_radians) / self.state_step)
        self.within_trial_arrays['code_received_t'][self.within_trial_params["time_ind"]] = np.random.choice(np.arange(self.num_observations), p=self.emission_matrix_true[:,int(state_visited)])

        print(f'Current angle: {self.within_trial_arrays["angles_visited"][self.within_trial_params["time_ind"]]}')
        print(f'Code received: {self.within_trial_arrays["code_received_t"][self.within_trial_params["time_ind"]]}')

    def bayes_update(self, numerator):
        denominator = numerator.sum()
        if denominator > 0:
            return numerator / denominator
        else:
            return numerator
        
    def update_likelihood_given_observed_sequence(self, obs_sequence_including_forecast_from_targets, obs_sequence_excluding_forecast_from_targets):
        if (obs_sequence_excluding_forecast_from_targets>0).all():
            likelihood_obs_forecast_t_given_received_obs = obs_sequence_including_forecast_from_targets / obs_sequence_excluding_forecast_from_targets
        else:
            likelihood_obs_forecast_t_given_received_obs = np.zeros(self.num_beliefs)
            for target_t in range(self.num_beliefs):
                obs_sequence_including_forecast_from_target_t = obs_sequence_including_forecast_from_targets[target_t]
                obs_sequence_excluding_forecast_from_target_t = obs_sequence_excluding_forecast_from_targets[target_t]
                if (obs_sequence_excluding_forecast_from_target_t > 0):
                    likelihood_obs_from_target_forecast_t_given_received_obs_fromtarget = obs_sequence_including_forecast_from_target_t / obs_sequence_excluding_forecast_from_target_t
                else:
                    likelihood_obs_from_target_forecast_t_given_received_obs_fromtarget = obs_sequence_including_forecast_from_target_t
                likelihood_obs_forecast_t_given_received_obs[target_t] = likelihood_obs_from_target_forecast_t_given_received_obs_fromtarget
        
        return likelihood_obs_forecast_t_given_received_obs
        
    def update_posterior_and_compute_current_entropy(self):
        if self.within_trial_params["time_ind"]<1:
            self.within_trial_arrays["joint_prob_received_obs_state_t_from_targets"][self.within_trial_params["time_ind"]] = self.prior_state_prob * self.emission_matrix_targets[:,int(self.within_trial_arrays['code_received_t'][self.within_trial_params["time_ind"]])]
        else:
            state_transition_matrix = self.get_deterministic_state_transition_matrix(self.within_trial_arrays['steps_taken'][int(self.within_trial_params["time_ind"]-1)])
            prob_of_received_obs_from_states_given_targets = self.emission_matrix_targets[:,int(self.within_trial_arrays['code_received_t'][self.within_trial_params["time_ind"]])]
            joint_prob_of_previous_obs_from_states = self.within_trial_arrays['joint_prob_received_obs_state_t_from_targets'][self.within_trial_params["time_ind"]-1]
            self.within_trial_arrays['joint_prob_received_obs_state_t_from_targets'][self.within_trial_params["time_ind"]] = (joint_prob_of_previous_obs_from_states @ state_transition_matrix) * prob_of_received_obs_from_states_given_targets

        self.within_trial_arrays['likelihood_received_obs_from_targets_t'][self.within_trial_params["time_ind"]] = self.within_trial_arrays['joint_prob_received_obs_state_t_from_targets'][self.within_trial_params["time_ind"]].sum(axis=1)

        if self.within_trial_params["time_ind"]<1:
            likelihood_received_obs = self.within_trial_arrays['likelihood_received_obs_from_targets_t'][self.within_trial_params["time_ind"]]
            print(self.prior_target_prob*likelihood_received_obs)
            self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]] = self.bayes_update(self.prior_target_prob*likelihood_received_obs)
        else:
            likelihood_received_obs_given_observed_sequence = self.update_likelihood_given_observed_sequence(self.within_trial_arrays['likelihood_received_obs_from_targets_t'][self.within_trial_params["time_ind"]], self.within_trial_arrays['likelihood_received_obs_from_targets_t'][self.within_trial_params["time_ind"]-1])
            self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]] = self.bayes_update(self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]-1]*likelihood_received_obs_given_observed_sequence)

        self.within_trial_arrays['current_entropyS_t'][self.within_trial_params["time_ind"]] = self.compute_entropy_from_posterior_across_targets(self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]])

    def debug_within_trial_print_statements(self):
        for target_t in range(self.num_beliefs):
            if int(self.within_trial_params["time_ind"]+1)>1:
                obs_string = f'O1:{int(self.within_trial_params["time_ind"]+1)}={self.within_trial_arrays['code_received_t'][:int(self.within_trial_params["time_ind"]+1)]}'
            else:
                obs_string = f'O1={self.within_trial_arrays['code_received_t'][:int(self.within_trial_params["time_ind"]+1)]}'

            if int(self.within_trial_params["time_ind"]+1)>2:
                angle_steps_string = f'Δθ2:{int(self.within_trial_params["time_ind"]+1)}'
            else:
                angle_steps_string = f'Δθ2'

            joint_prob_term_val = self.within_trial_arrays["joint_prob_received_obs_state_t_from_targets"][self.within_trial_params["time_ind"], target_t]
            if int(self.within_trial_params["time_ind"]+1)>1:
                joint_prob_term_name = f'P({obs_string},S{int(self.within_trial_params["time_ind"]+1)}|{angle_steps_string},T{int(target_t+1)})'
            else:
                joint_prob_term_name = f'P({obs_string},S{int(self.within_trial_params["time_ind"]+1)}|T{int(target_t+1)})'
            joint_prob_string = f'{joint_prob_term_name}={joint_prob_term_val.round(4)}'

            likelihood_of_obs_term_val = self.within_trial_arrays["likelihood_received_obs_from_targets_t"][self.within_trial_params["time_ind"], target_t]
            if int(self.within_trial_params["time_ind"]+1)>1:
                likelihood_of_obs_term_name = f'P({obs_string}|{angle_steps_string},T{int(target_t+1)})'
            else:
                likelihood_of_obs_term_name = f'P({obs_string}|T{int(target_t+1)})'
            likelihood_of_obs_string = f'{likelihood_of_obs_term_name}={likelihood_of_obs_term_val}'

            if int(self.within_trial_params["time_ind"]+1)>1:
                if likelihood_of_obs_term_val>0:
                    state_belief_term_name = f'P(S{int(self.within_trial_params["time_ind"]+1)}|{obs_string},{angle_steps_string},T{int(target_t+1)})'
                    state_belief_string = f'{state_belief_term_name}={(joint_prob_term_val/likelihood_of_obs_term_val).round(2)}'
                else:
                    state_belief_term_name = f'P(S{int(self.within_trial_params["time_ind"]+1)}|{obs_string},{angle_steps_string},T{int(target_t+1)})'
                    state_belief_string = f'{state_belief_term_name}={(joint_prob_term_val).round(2)}'
            else:
                if likelihood_of_obs_term_val>0:
                    state_belief_term_name = f'P(S{int(self.within_trial_params["time_ind"]+1)}|{obs_string},T{int(target_t+1)})'
                    state_belief_string = f'{state_belief_term_name}={(joint_prob_term_val/likelihood_of_obs_term_val).round(2)}'
                else:
                    state_belief_term_name = f'P(S{int(self.within_trial_params["time_ind"]+1)}|{obs_string},T{int(target_t+1)})'
                    state_belief_string = f'{state_belief_term_name}={(joint_prob_term_val).round(2)}'
            
            if int(self.within_trial_params["time_ind"]+1)>1:
                target_belief_term_name = f'P(T|{obs_string},{angle_steps_string})'
            else:
                target_belief_term_name = f'P(T|{obs_string})'
            target_belief_string = f'{target_belief_term_name}={self.within_trial_arrays["posterior_t"][self.within_trial_params["time_ind"]]}'

            current_entropy_term_name = f'H{int(self.within_trial_params["time_ind"]+1)}'
            current_entropy_string = f'{current_entropy_term_name}={self.within_trial_arrays["current_entropyS_t"][self.within_trial_params["time_ind"]]:.2f}'
            print(f'{joint_prob_string}, {likelihood_of_obs_string}, {state_belief_string}, {target_belief_string}, {current_entropy_string}')

    def debug_forecasting_each_obs_print_statements(self):
        likelihood_obs_from_target_forecast_t_fromtargets = self.within_trial_arrays['joint_prob_received_obs_state_forecast_t_fromtargets'][self.within_trial_params["time_ind"]].sum(axis=1)
        forecasted_obs_string = f'O{int(self.within_trial_params["time_ind"]+2)}={self.within_trial_arrays['forecasted_obs']}'

        if int(self.within_trial_params["time_ind"]+1)>1:
            up_to_date_obs_sequence_term_name = f'O1:{int(self.within_trial_params["time_ind"]+1)}'
        else:
            up_to_date_obs_sequence_term_name = f'O1'
        up_to_date_obs_sequence_string = f'{up_to_date_obs_sequence_term_name}={self.within_trial_arrays['code_received_t'][:int(self.within_trial_params["time_ind"]+1)]}'

        if int(self.within_trial_params["time_ind"]+1)>1:
            angle_steps_string = f'Δθ2:{int(self.within_trial_params["time_ind"]+1)}'
        else:
            angle_steps_string = f'Δθ2'

        print(f'P({forecasted_obs_string},{up_to_date_obs_sequence_string}|{angle_steps_string},T)={likelihood_obs_from_target_forecast_t_fromtargets}')
        print(f'P({forecasted_obs_string}|{up_to_date_obs_sequence_string},{angle_steps_string},T)={self.within_trial_arrays['likelihood_obs_forecast_t_given_received_obs']}')

    def compute_entropy_from_posterior_across_targets(self, posterior):
        forecasted_S_from_obs = 0
        for target_t in range(self.num_beliefs):
            if posterior[target_t] > 0:
                forecasted_S_from_obs += (-posterior[target_t] * np.log2(posterior[target_t]))
        return forecasted_S_from_obs

    def compute_forecasted_entropy_from_forecasted_obs(self, state_transition_matrix):
        forecasted_S = np.zeros(self.num_observations)
        prob_obs_at_forecast_t_given_obs_at_cur_t = np.zeros(self.num_observations)
        posterior_forecast_given_received_obs = np.zeros((self.num_observations, self.num_beliefs))
        for obs in range(self.num_observations):
            self.within_trial_arrays['joint_prob_received_obs_state_forecast_t_fromtargets'][self.within_trial_params["time_ind"]] = (self.within_trial_arrays['joint_prob_received_obs_state_t_from_targets'][self.within_trial_params["time_ind"]] @ state_transition_matrix) * self.emission_matrix_targets[:,obs]
            likelihood_obs_from_target_forecast_t_fromtargets = self.within_trial_arrays['joint_prob_received_obs_state_forecast_t_fromtargets'][self.within_trial_params["time_ind"]].sum(axis=1)
            likelihood_obs_forecast_t_given_received_obs = self.update_likelihood_given_observed_sequence(likelihood_obs_from_target_forecast_t_fromtargets, self.within_trial_arrays['likelihood_received_obs_from_targets_t'][self.within_trial_params["time_ind"]])

            self.within_trial_arrays['forecasted_obs'] = obs
            self.within_trial_arrays['likelihood_obs_forecast_t_given_received_obs'] = likelihood_obs_forecast_t_given_received_obs
            self.debug_forecasting_each_obs_print_statements()
            
            numerator = (self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]] * likelihood_obs_forecast_t_given_received_obs)
            prob_obs_at_forecast_t_given_obs_at_cur_t[obs] = numerator.sum()
            posterior_forecast_given_received_obs[obs] = self.bayes_update(numerator)
            forecasted_S[obs] = self.compute_entropy_from_posterior_across_targets(posterior_forecast_given_received_obs[obs])

        return prob_obs_at_forecast_t_given_obs_at_cur_t, forecasted_S
    
    def compute_expected_entropy_change_from_each_action(self, print_debug=True):
        self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]] = self.within_trial_arrays['current_entropyS_t'][self.within_trial_params["time_ind"]] 
        self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]] -= self.within_trial_arrays['expected_forecast_t_S_per_step'][self.within_trial_params["time_ind"]]

        if print_debug:
            print(f'S{int(self.within_trial_params["time_ind"]+1)}={self.within_trial_arrays['current_entropyS_t'][self.within_trial_params["time_ind"]]}') 
            print(f'S{int(self.within_trial_params["time_ind"]+2)}={self.within_trial_arrays['expected_forecast_t_S_per_step'][self.within_trial_params["time_ind"]]}')
            print(f'ΔS{int(self.within_trial_params["time_ind"]+1)}→{int(self.within_trial_params["time_ind"]+2)}={self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]]}')

    def print_debug_forecasted_entropy_statements(self, prob_obs_at_forecast_t_given_obs_at_cur_t, forecasted_S):
        expected_forecast_t_S = (forecasted_S * prob_obs_at_forecast_t_given_obs_at_cur_t).sum()
        print_string = f'S{int(self.within_trial_params["time_ind"]+2)} = '
        for obs in range(self.num_observations):
            if obs < self.num_observations - 1:
                print_string += f'{prob_obs_at_forecast_t_given_obs_at_cur_t[obs]:.3f}x{forecasted_S[obs]:.3f} + '
            else:
                print_string += f'{prob_obs_at_forecast_t_given_obs_at_cur_t[obs]:.3f}x{forecasted_S[obs]:.3f}'
        print_string += f' = {expected_forecast_t_S}'
        print(print_string)

    def forecast_and_compute_expected_entropy_from_actions(self):
        for i, forecast_t_candidate_step in enumerate(self.candidate_steps):
            state_transition_matrix = (self.get_deterministic_state_transition_matrix(forecast_t_candidate_step))
            prob_obs_at_forecast_t_given_obs_at_cur_t, forecasted_S = self.compute_forecasted_entropy_from_forecasted_obs(state_transition_matrix)
            expected_forecast_t_S = (forecasted_S * prob_obs_at_forecast_t_given_obs_at_cur_t).sum()
            self.print_debug_forecasted_entropy_statements(prob_obs_at_forecast_t_given_obs_at_cur_t, forecasted_S)
            self.within_trial_arrays['expected_forecast_t_S_per_step'][self.within_trial_params["time_ind"], i] = expected_forecast_t_S

    def determine_decision_from_forecasted_entropy_change(self):
        unique, counts = np.unique(self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]], return_counts=True)
        frequent_cond = counts>1
        unique_frequent_delta_S = unique[frequent_cond]
        if (counts[unique==self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]].max()]==1).all():
            print('Choosing according to max ΔS')
            self.within_trial_arrays['decision_type_t'][self.within_trial_params["time_ind"]] = 'max ΔS'
            step_to_take = self.candidate_steps[self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]].argmax()]
        else:
            print(f'DETERMINING DECISION {self.within_trial_arrays["posterior_t"][self.within_trial_params["time_ind"]]}: {(1 in self.within_trial_arrays["posterior_t"][self.within_trial_params["time_ind"]])}')
            if (1 not in self.within_trial_arrays['posterior_t'][self.within_trial_params["time_ind"]]):
                print(f'Choosing randomly between {self.candidate_steps[np.where(self.within_trial_arrays["deltaS_t"][self.within_trial_params["time_ind"]]==unique_frequent_delta_S.max())[0]]}')
                self.within_trial_arrays['decision_type_t'][self.within_trial_params["time_ind"]] = 'random'
                step_to_take = self.candidate_steps[np.random.choice(np.where(self.within_trial_arrays['deltaS_t'][self.within_trial_params["time_ind"]]==unique_frequent_delta_S.max())[0])]
            else:
                print('DECISION REACHED')
                self.within_trial_arrays['decision_type_t'][self.within_trial_params["time_ind"]] = 'end'
                step_to_take = 0

        self.within_trial_arrays['steps_taken'][self.within_trial_params["time_ind"]] = step_to_take

    def get_summary_of_trial(self):
        trial_save_variables = ['posterior_t', 'expected_forecast_t_S_per_step', 'angles_visited', 'steps_taken', 'current_entropyS_t', 'code_received_t', 'decision_type_t', 'time_taken']
        summary_df = pd.DataFrame()
        for sav_var in trial_save_variables:
            if sav_var == 'time_taken':
                chunk = self.within_trial_params['time_ind']*np.ones(self.within_trial_params['time_ind'])
                summary_df = pd.concat([summary_df, pd.DataFrame(chunk, columns=[sav_var])], axis=1)
            else:
                chunk = self.within_trial_arrays[sav_var][:self.within_trial_params['time_ind']]
                if len(chunk.shape)==1:
                    summary_df = pd.concat([summary_df, pd.DataFrame(chunk, columns=[sav_var])], axis=1)
                else:
                    if sav_var=='posterior_t':
                        column_names = []
                        for tar in range(chunk.shape[1]):
                            column_names += [f'{sav_var}_T{int(tar)+1}']
                        summary_df = pd.concat([summary_df, pd.DataFrame(chunk, columns=column_names)], axis=1)
                    if sav_var=='expected_forecast_t_S_per_step':
                        summary_df = pd.concat([summary_df, pd.DataFrame(chunk, columns=[f'{sav_var}_CW', f'{sav_var}_STAY', f'{sav_var}_CCW'])], axis=1)
        return summary_df
    
    def get_deterministic_states_from_emission_matrix(self, emission_matrix):
        code_regions_info = dict()
        regions = np.array(np.where(emission_matrix[1:,:]==1))
        code_regions_info['states'] = regions[1,:]
        code_regions_info['observations'] = regions[0,:]+1
        code_regions_info['first_nonzero_state_center'] = self.state_centers[code_regions_info['states']][0]
        code_regions_info['last_nonzero_state_center'] = self.state_centers[code_regions_info['states']][-1]

        return code_regions_info

    def plot_code_dependent_regions(self, cur_ax, emission_matrix, setup_details):
        pie = np.pi
        if (~(emission_matrix[1:]==1)).all():
            code_received = 0
            theta_obs0 = np.linspace(0, 2*pie, 30)
            x_inner = setup_details['inner_radius'] * np.cos(theta_obs0)
            y_inner =  setup_details['inner_radius'] * np.sin(theta_obs0)
            x_poly = np.concatenate([setup_details['outer_radius']*x_inner, x_inner[::-1]])
            y_poly = np.concatenate([setup_details['outer_radius']*y_inner, y_inner[::-1]])
            cur_ax.fill(x_poly, y_poly, facecolor='orange', alpha=0.25, linewidth=0)
            cur_ax.plot(x_inner, y_inner, color='orange', linewidth=1)
        else:
            code_regions_info = self.get_deterministic_states_from_emission_matrix(emission_matrix)
            theta_obs0 = np.linspace(0, code_regions_info['first_nonzero_state_center'] - (self.state_step/2), 30)
            x_inner = setup_details['inner_radius'] * np.cos(theta_obs0)
            y_inner =  setup_details['inner_radius'] * np.sin(theta_obs0)
            x_poly = np.concatenate([setup_details['outer_radius']*x_inner, x_inner[::-1]])
            y_poly = np.concatenate([setup_details['outer_radius']*y_inner, y_inner[::-1]])
            cur_ax.fill(x_poly, y_poly, facecolor='orange', alpha=0.25, linewidth=0)
            cur_ax.plot(x_inner, y_inner, color='orange', linewidth=1)

            code_received = 0
            for i, state_c in enumerate(self.state_centers[code_regions_info['states']]):
                if setup_details['agent_radians'] > (state_c - (self.state_step/2)) and setup_details['agent_radians'] < (state_c + (self.state_step/2)):
                    code_received = code_regions_info['observations'][i]
                theta_obs_not0 = np.linspace(state_c - (self.state_step/2), state_c + (self.state_step/2), 30)
                x_inner = setup_details['inner_radius'] * np.cos(theta_obs_not0)
                y_inner =  setup_details['inner_radius'] * np.sin(theta_obs_not0)
                x_poly = np.concatenate([setup_details['outer_radius']*x_inner, x_inner[::-1]])
                y_poly = np.concatenate([setup_details['outer_radius']*y_inner, y_inner[::-1]])
                cur_ax.fill(x_poly, y_poly, facecolor=self.color_code_map[code_regions_info['observations'][i]], alpha=0.25, linewidth=0)
                cur_ax.plot(x_inner, y_inner, color=self.color_code_map[code_regions_info['observations'][i]], linewidth=1)
                
                if i < code_regions_info['states'].shape[0]-1:
                    theta_obs0 = np.linspace(state_c + (self.state_step/2), self.state_centers[code_regions_info['states']][i+1] - (self.state_step/2), 30)
                    x_inner = setup_details['inner_radius'] * np.cos(theta_obs0)
                    y_inner =  setup_details['inner_radius'] * np.sin(theta_obs0)
                    x_poly = np.concatenate([setup_details['outer_radius']*x_inner, x_inner[::-1]])
                    y_poly = np.concatenate([setup_details['outer_radius']*y_inner, y_inner[::-1]])
                    cur_ax.fill(x_poly, y_poly, facecolor='orange', alpha=0.25, linewidth=0)
                    cur_ax.plot(x_inner, y_inner, color='orange', linewidth=1)

            theta_obs0 = np.linspace(code_regions_info['last_nonzero_state_center'] + (self.state_step/2), 2*pie, 30)
            x_inner = setup_details['inner_radius'] * np.cos(theta_obs0)
            y_inner =  setup_details['inner_radius'] * np.sin(theta_obs0)
            x_poly = np.concatenate([setup_details['outer_radius']*x_inner, x_inner[::-1]])
            y_poly = np.concatenate([setup_details['outer_radius']*y_inner, y_inner[::-1]])
            cur_ax.fill(x_poly, y_poly, facecolor='orange', alpha=0.25, linewidth=0)
            cur_ax.plot(x_inner, y_inner, color='orange', linewidth=1)

        return code_received

    def plot_agent_in_env(self, cur_ax, setup_details, code_received):
        for theta in self.state_centers:
            r_outer = 4 * setup_details['outer_radius']
            cur_ax.plot([setup_details['inner_radius']*np.cos(theta), r_outer*np.cos(theta)], 
                    [setup_details['inner_radius']*np.sin(theta), r_outer*np.sin(theta)],
                        color='k', linewidth=1.5, solid_capstyle='round', zorder=1)
            
        initial_state = np.floor((setup_details['agent_radians'] + (self.state_step/2)) / self.state_step) % self.num_states
        cur_ax.scatter(setup_details['outer_radius']*np.cos(self.state_centers[int(initial_state)]), setup_details['outer_radius']*np.sin(self.state_centers[int(initial_state)]),
                    s=200, marker='.', facecolor='r', edgecolor='k', zorder=4, label=f'State={initial_state}\nCode={code_received}')
        cur_ax.set_title(f'{setup_details["title"]} Model')
        cur_ax.set_aspect('equal', adjustable='box')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        cur_ax.set_xlim(-setup_details['grid_extent'], setup_details['grid_extent'])
        cur_ax.set_ylim(-setup_details['grid_extent'], setup_details['grid_extent'])
        cur_ax.legend()

    def plot_agent_env_setup(self, cur_ax, emission_matrix, setup_details):
        code_received = self.plot_code_dependent_regions(cur_ax, emission_matrix, setup_details)
        self.plot_agent_in_env(cur_ax, setup_details, code_received)

    def observe_angle_state_pair(self, initial_angle):
        initial_radians = np.radians((initial_angle % 360))

        fig, ax = plt.subplots(1, self.num_beliefs, figsize=(6*self.num_beliefs, 6))
        setup_details = {'title': 'Target', 'grid_extent': 10, 
                            'inner_radius' : 1, 'outer_radius' : 5,
                            'agent_radians' : initial_radians}
        for target_t in range(self.num_beliefs):
            setup_details['title'] = f'Target {int(target_t+1)}'
            self.plot_agent_env_setup(ax[target_t], self.emission_matrix_targets[target_t], setup_details)        
        plt.show()
