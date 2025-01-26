import numpy as np
from model import Model

config = {
    "Rmax": 1,
    "Rmax_k": 50,
    "Rmax_max_iterations": 2000
}

def r_max(env, policy_function,
          Rmax=config["Rmax"],
          k=config["Rmax_k"],
          max_iterations=config["Rmax_max_iterations"]):

    # Initialize enhanced model with optimistic state
    enhanced_model = Model(num_states=env.observation_space.n + 1, num_actions=env.action_space.n)

    # Configure optimistic state properties
    optimistic_state = env.observation_space.n
    enhanced_model.rewards[optimistic_state, :] = Rmax
    enhanced_model.transition_probs[:, :, optimistic_state] = 1

    # Initialize learning statistics
    state_action_transitions = np.zeros((enhanced_model.num_states, enhanced_model.num_actions, enhanced_model.num_states), dtype=int)
    reward_history = np.zeros((enhanced_model.num_states, enhanced_model.num_actions, enhanced_model.num_states), dtype=int)
    visit_counter = np.zeros((enhanced_model.num_states, enhanced_model.num_actions), dtype=int)
    experienced_pairs = np.full((enhanced_model.num_states, enhanced_model.num_actions), False, dtype=bool)
    discovered_count = 0

    # Main learning loop
    for iteration in range(max_iterations):
        knowledge_stable = True
        current_policy = policy_function(enhanced_model)

        # Execute policy and gather experience
        current_state = env.reset()[0]
        while knowledge_stable:
            chosen_action = current_policy[current_state]
            next_state, immediate_reward, terminal, _, _ = env.step(chosen_action)

            # Update transition statistics
            state_action_transitions[current_state, chosen_action, next_state] += 1
            reward_history[current_state, chosen_action, next_state] = immediate_reward
            visit_counter[current_state, chosen_action] += 1

            # Check for newly experienced state-action pairs
            if not experienced_pairs[current_state, chosen_action] and \
               visit_counter[current_state, chosen_action] >= k:
                experienced_pairs[current_state, chosen_action] = True
                discovered_count += 1
                knowledge_stable = False
            else:
                current_state = next_state if not experienced_pairs[current_state, chosen_action] \
                              or not terminal else env.reset()[0]

        # Update model estimates
        for state in range(enhanced_model.num_states):
            for action in range(enhanced_model.num_actions):
                if experienced_pairs[state, action]:
                    enhanced_model.rewards[state, action] = 0
                    for next_state in range(enhanced_model.num_states):
                        transition_prob = state_action_transitions[state, action, next_state] / \
                                        visit_counter[state, action]
                        enhanced_model.transition_probs[state, action, next_state] = transition_prob
                        enhanced_model.rewards[state, action] += reward_history[state, action, next_state] * \
                                                               transition_prob

        # Check learning completion
        if discovered_count >= (enhanced_model.num_states - 1) * enhanced_model.num_actions:
            break

    # Extract final policy
    final_policy = policy_function(enhanced_model)
    return final_policy[:-1]  # Exclude optimistic state from final policy