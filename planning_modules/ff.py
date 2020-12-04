from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException

from pddlgym.parser import parse_plan_step
from settings import AgentConfig as ac

import sys
import os
import re
import subprocess
import time



class FastForwardPlanner(Planner):
    FF_PATH = os.environ['FF_PATH']

    def get_policy(self, raw_problem_fname):
        actions = self.get_plan(raw_problem_fname)
        def policy(_):
            if len(actions) == 0:
                raise NoPlanFoundException() 
            return actions.pop(0)
        return policy

    def get_plan(self, raw_problem_fname, use_cache=True):
        # If there are no operators yet, we're not going to be able to find a plan
        if not self._learned_operators:
            raise NoPlanFoundException()
        domain_fname = self._create_domain_file()
        problem_fname, objects = self._create_problem_file(raw_problem_fname, use_cache=use_cache)
        cmd_str = self._get_cmd_str(domain_fname, problem_fname)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str)
        end_time = time.time()
        os.remove(domain_fname)
        if not use_cache:
            os.remove(problem_fname)
        if end_time - start_time > 0.9*ac.planner_timeout:
            raise PlannerTimeoutException()
        plan = self._output_to_plan(output)
        actions = self._plan_to_actions(plan, objects)
        return actions

    def _get_cmd_str(self, domain_fname, problem_fname):
        timeout = "gtimeout" if sys.platform == "darwin" else "timeout"
        return "{} {} {} -o {} -f {}".format(
            timeout, ac.planner_timeout, self.FF_PATH,
            domain_fname, problem_fname)

    @staticmethod
    def _output_to_plan(output):
        if not output.strip() or \
           "goal can be simplified to FALSE" in output or \
            "unsolvable" in output:
            raise NoPlanFoundException()
        plan = re.findall(r"\d+?: (.+)", output.lower())
        if not plan and "found legal" not in output and \
           "The empty plan solves it" not in output:
            raise Exception("Plan not found with FF! Error: {}".format(output))
        return plan

    def _plan_to_actions(self, plan, objects):
        operators = self._learned_operators
        action_predicates = self._action_space.predicates

        actions = []
        for plan_step in plan:
            if plan_step == "reach-goal":
                continue
            action = parse_plan_step(plan_step, operators, action_predicates, objects)
            actions.append(action)
        return actions
