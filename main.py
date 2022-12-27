"""Top-level script for learning operators.
"""
import matplotlib
matplotlib.use("Agg")
from agent import Agent
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from plotting import plot_results
from settings import AgentConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc

from collections import defaultdict

import glob
import time
import gym
import numpy as np
import os
import pddlgym
import pickle


class Runner:
    """Helper class for running experiments.
    """
    def __init__(self, agent, train_env, test_env, domain_name, curiosity_name):
        self.agent = agent
        self.train_env = train_env
        self.test_env = test_env
        self.domain_name = domain_name
        self.curiosity_name = curiosity_name
        self.num_train_iters = ac.num_train_iters[domain_name]
        self._variational_dist_transitions = self._initialize_variational_distance_transitions()

    def _initialize_variational_distance_transitions(self):
        print("Getting transitions for variational distance...")
        fname = "data/{}.vardisttrans".format(self.domain_name)
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                transitions = pickle.load(f)
            return transitions
        actions = self.test_env.action_space.predicates
        total_counts = {a: 0 for a in actions}
        num_no_effects = {a: 0 for a in actions}
        transitions = []
        num_trans_per_act = 100
        if self.domain_name in ec.num_test_problems:
            num_problems = ec.num_test_problems[self.domain_name]
        else:
            num_problems = len(self.test_env.problems)
        while True:
            if all(c >= num_trans_per_act for c in total_counts.values()):
                break
            obs, _ = self.test_env.reset()
            for _1 in range(ec.num_var_dist_trans[self.domain_name]//num_problems):
                action = self.test_env.action_space.sample(obs)
                next_obs, _, done, _ = self.test_env.step(action)
                null_effect = (next_obs.literals == obs.literals)
                keep_transition = ((not null_effect or
                                    (num_no_effects[action.predicate] <
                                     total_counts[action.predicate]/2+1)) and
                                   total_counts[action.predicate] < num_trans_per_act)
                if keep_transition:
                    total_counts[action.predicate] += 1
                    if null_effect:
                        num_no_effects[action.predicate] += 1
                    transitions.append((obs, action, next_obs))
                if done:
                    break
                obs = next_obs
        with open(fname, "wb") as f:
            pickle.dump(transitions, f)
        return transitions

    def run(self):
        """Run primitive operator learning loop.
        """
        results = []
        episode_done = True
        episode_time_step = 0
        itrs_on = None
        for itr in range(self.num_train_iters):
            if gc.verbosity > 0:
                print("\nIteration {} of {}".format(itr, self.num_train_iters))

            # Gather training data
            if gc.verbosity > 2:
                print("Gathering training data...")

            if episode_done or episode_time_step > ac.max_train_episode_length[self.domain_name]:
                obs, _ = self.train_env.reset()
                self.agent.reset_episode(obs)
                episode_time_step = 0

            action = self.agent.get_action(obs)

            next_obs, _, episode_done, _ = self.train_env.step(action)
            self.agent.observe(obs, action, next_obs)
            obs = next_obs
            episode_time_step += 1

            # Learn and test
            if itr % ac.learning_interval[self.domain_name] == 0:
                start = time.time()
                if gc.verbosity > 1:
                    print("Learning...")

                if self.domain_name == "PybulletBlocks" and self.curiosity_name == "oracle":
                    operators_changed = True
                else:
                    operators_changed = self.agent.learn()

                # Only rerun tests if operators have changed, or stochastic env
                if operators_changed or ac.planner_name[self.domain_name] == "ffreplan" or \
                   itr + ac.learning_interval[self.domain_name] >= self.num_train_iters:  # last:
                    start = time.time()
                    if gc.verbosity > 1:
                        print("Testing...")

                    test_solve_rate, variational_dist = self._evaluate_operators()

                    if gc.verbosity > 1:
                        print("Result:", test_solve_rate, variational_dist)
                        print("Testing took {} seconds".format(time.time()-start))

                    if "oracle" in self.agent.curiosity_module_name and \
                       test_solve_rate == 1 and ac.planner_name[self.domain_name] == "ff":
                        # Oracle can be done when it reaches 100%, if deterministic env
                        self.agent._curiosity_module.turn_off()
                        self.agent._operator_learning_module.turn_off()
                        if itrs_on is None:
                            itrs_on = itr

                else:
                    assert results, "operators_changed is False but never learned any operators..."
                    if gc.verbosity > 1:
                        print("No operators changed, continuing...")

                    test_solve_rate = results[-1][1]
                    variational_dist = results[-1][2]
                    if gc.verbosity > 1:
                        print("Result:", test_solve_rate, variational_dist)
                results.append((itr, test_solve_rate, variational_dist))

        if itrs_on is None:
            itrs_on = self.num_train_iters
        curiosity_avg_time = self.agent.curiosity_time/itrs_on

        return results, curiosity_avg_time

    def _evaluate_operators(self):
        """Test current operators. Return (solve rate on test suite,
        average variational distance).
        """
        for op in self.agent.learned_operators:
            print(op)
        if self.domain_name == "PybulletBlocks" and self.curiosity_name == "oracle":
            # Disable oracle for pybullet.
            return 0.0, 1.0
        num_successes = 0
        if self.domain_name in ec.num_test_problems:
            num_problems = ec.num_test_problems[self.domain_name]
        else:
            num_problems = len(self.test_env.problems)
        for problem_idx in range(num_problems):
            print("\tTest case {} of {}, {} successes so far".format(
                problem_idx+1, num_problems, num_successes), end="\r")
            self.test_env.fix_problem_index(problem_idx)
            obs, debug_info = self.test_env.reset()
            try:
                policy = self.agent.get_policy(debug_info["problem_file"])
            except (NoPlanFoundException, PlannerTimeoutException):
                # Automatic failure
                continue
            # Test plan open-loop
            reward = 0.
            for _ in range(ac.max_test_episode_length[self.domain_name]):
                try:
                    action = policy(obs)
                except (NoPlanFoundException, PlannerTimeoutException):
                    break
                obs, reward, done, _ = self.test_env.step(action)
                if done:
                    break
            # Reward is 1 iff goal is reached
            if reward == 1.:
                num_successes += 1
            else:
                assert reward == 0.
        print()
        variational_dist = 0
        for state, action, next_state in self._variational_dist_transitions:
            if ac.learning_name.startswith("groundtruth"):
                predicted_next_state = self.agent._curiosity_module._get_predicted_next_state_ops(state, action)
            else:
                predicted_next_state = self.agent._curiosity_module.sample_next_state(state, action)
            if predicted_next_state is None or \
               predicted_next_state.literals != next_state.literals:
                variational_dist += 1
        variational_dist /= len(self._variational_dist_transitions)
        return float(num_successes)/num_problems, variational_dist

def _run_single_seed(seed, domain_name, curiosity_name, learning_name):
    start = time.time()

    ac.seed = seed
    ec.seed = seed
    ac.planner_timeout = 60 if "oracle" in curiosity_name else 10

    train_env = gym.make("PDDLEnv{}-v0".format(domain_name))
    train_env.seed(ec.seed)
    # MAJOR HACK. Only used by oracle_curiosity.py and by the LLM-based
    # learner, which uses the environment to access the predicates and
    # action names.
    ac.train_env = train_env
    agent = Agent(domain_name, train_env.action_space,
                  train_env.observation_space, curiosity_name, learning_name,
                  planning_module_name=ac.planner_name[domain_name])
    test_env = gym.make("PDDLEnv{}Test-v0".format(domain_name))
    results, curiosity_avg_time = Runner(agent, train_env, test_env, domain_name, curiosity_name).run()
    with open("results/timings/{}_{}_{}_{}.txt".format(domain_name, curiosity_name, learning_name, seed), "w") as f:
        f.write("{} {} {} {} {}\n".format(domain_name, curiosity_name, learning_name, seed, curiosity_avg_time))

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", domain_name, learning_name, curiosity_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    cache_file = os.path.join(outdir, "{}_{}_{}_{}.pkl".format(
        domain_name, learning_name, curiosity_name, seed))
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
        print("Dumped results to {}".format(cache_file))
    print("\n\n\nFinished single seed in {} seconds".format(time.time()-start))
    return {curiosity_name: results}


def _main():
    start = time.time()
    if not os.path.exists("results/"):
        os.mkdir("results/")
    if not os.path.exists("results/timings/"):
        os.mkdir("results/timings/")
    if not os.path.exists("data/"):
        os.mkdir("data/")

    if isinstance(ec.domain_name, str):
        ec.domain_name = [ec.domain_name]

    for domain_name in ec.domain_name:
        all_results = defaultdict(list)
        for curiosity_name in ac.curiosity_methods_to_run:
            if curiosity_name in ac.cached_results_to_load:
                for pkl_fname in glob.glob(os.path.join(
                        "results/", domain_name, ac.learning_name,
                        curiosity_name, "*.pkl")):
                    with open(pkl_fname, "rb") as f:
                        saved_results = pickle.load(f)
                    all_results[curiosity_name].append(saved_results)
                if curiosity_name not in all_results:
                    print("WARNING: Found no results to load for {}".format(
                        curiosity_name))
            else:
                for seed in range(gc.num_seeds):
                    seed = seed+20
                    print("\nRunning curiosity method: {}, with seed: {}\n".format(
                        curiosity_name, seed))
                    single_seed_results = _run_single_seed(
                        seed, domain_name, curiosity_name, ac.learning_name)
                    for cur_name, results in single_seed_results.items():
                        all_results[cur_name].append(results)
                    plot_results(domain_name, ac.learning_name, all_results)
                    plot_results(domain_name, ac.learning_name, all_results, dist=True)

        plot_results(domain_name, ac.learning_name, all_results)
        plot_results(domain_name, ac.learning_name, all_results, dist=True)

    print("\n\n\n\n\nFinished in {} seconds".format(time.time()-start))


if __name__ == "__main__":
    _main()
