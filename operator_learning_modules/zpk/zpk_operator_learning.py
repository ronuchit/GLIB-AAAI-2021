import numpy as np
from settings import AgentConfig as ac
from pddlgym.parser import PDDLDomainParser
from pddlgym.structs import TypedEntity, ground_literal
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import get_transition_likelihood, print_rule_set, iter_variable_names
from ndr.ndrs import NOISE_OUTCOME, NDR, NDRSet
import openai

from collections import defaultdict
import tempfile


class ZPKOperatorLearningModule:

    def __init__(self, learned_operators, domain_name):
        self._domain_name = domain_name
        self._learned_operators = learned_operators
        self._transitions = defaultdict(list)
        self._seed = ac.seed
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._learning_on = True
        self._ndrs = {}
        self._fits_all_data = defaultdict(bool)

    def observe(self, state, action, effects):
        if not self._learning_on:
            return
        self._transitions[action.predicate].append((state.literals, action, effects))

        # Check whether we'll need to relearn
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False

    def learn(self):
        if not self._learning_on:
            return False

        # Check whether we have NDRs that need to be relearned
        is_updated = False
        for action_predicate in self._fits_all_data:
            if not self._fits_all_data[action_predicate]:
                transition_for_action = self._transitions[action_predicate]
                
                # This is used to prioritize samples in the learning batch
                def get_batch_probs(data):
                    assert False, "Assumed off"
                    # Favor more recent data
                    p = np.log(np.arange(1, len(data)+1)) + 1e-5
                    # Downweight empty transitions
                    for i in range(len(p)):
                        if len(data[i][2]) == 0:
                            p[i] /= 2.
                    p = p / p.sum()
                    return p

                # Initialize from previous set?
                if action_predicate in self._ndrs and \
                    ac.zpk_initialize_from_previous_rule_set[self._domain_name]:
                    init_rule_sets = {action_predicate : self._ndrs[action_predicate]}
                else:
                    init_rule_sets = None

                # max explain_examples_transitions
                max_ee_transitions = ac.max_zpk_explain_examples_transitions[self._domain_name]

                learned_ndrs = learn_ndrs({action_predicate : transition_for_action},
                    max_timeout=ac.max_zpk_learning_time,
                    max_action_batch_size=ac.max_zpk_action_batch_size[self._domain_name],
                    get_batch_probs=get_batch_probs,
                    init_rule_sets=init_rule_sets,
                    rng=self._rand_state,
                    max_ee_transitions=max_ee_transitions,
                )
                ndrs_for_action = learned_ndrs[action_predicate]
                self._ndrs[action_predicate] = ndrs_for_action
                self._fits_all_data[action_predicate] = True
                is_updated = True 

        # Update all learned_operators
        if is_updated:
            self._learned_operators.clear()
            for ndr_set in self._ndrs.values():
                for i, ndr in enumerate(ndr_set):
                    operator = ndr.determinize(name_suffix=i)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._learned_operators.add(operator)

            print_rule_set(self._ndrs)

        return is_updated

    def turn_off(self):
        self._learning_on = False

    def get_probability(self, transition):
        action = transition[1]
        if action.predicate not in self._ndrs:
            return 0.
        ndr_set = self._ndrs[action.predicate]
        selected_ndr = ndr_set.find_rule(transition)
        return get_transition_likelihood(transition, selected_ndr)

    def _ndr_fits_data(self, ndr, state, action, effects):
        prediction = ndr.predict_max(state.literals, action)
        return sorted(prediction) == sorted(effects)
        # return abs(1 - self.get_probability((state.literals, action, effects))) < 1e-5


class LLMZPKOperatorLearningModule(ZPKOperatorLearningModule):
    """The ZPK operator learner but initialized with operators output by an LLM."""

    def learn(self):
        # Initialize the operators from the LLM. Note that we do this in
        # learning rather than initialization because ac.train_env is created
        # after the agent is initialized in main.py.
        if not self._learned_operators:
            prompt = self._create_prompt()
            llm_output = self._query_llm(prompt)
            operators = self._llm_output_to_operators(llm_output)
            self._learned_operators.update(operators)
            # Also need to initialize ndrs!
            for op in operators:
                # In initializing the learner from previous, we assume a
                # standard variable naming scheme.
                action = [p for p in op.preconds.literals
                          if p.predicate in ac.train_env.action_space.predicates][0]
                preconditions = sorted(set(op.preconds.literals) - {action})
                effects = list(op.effects.literals)
                variables = list(action.variables)
                for lit in preconditions + op.effects.literals:
                    for v in lit.variables:
                        if v not in variables:
                            variables.append(v)
                sub = {old: TypedEntity(new_name, old.var_type)
                       for old, new_name in zip(variables, iter_variable_names())}
                action = ground_literal(action, sub)
                preconditions = [ground_literal(l, sub) for l in preconditions]
                effects = [ground_literal(l, sub) for l in effects]
                ndr = NDR(action, preconditions, np.array([1.0, 0.0]), [effects, [NOISE_OUTCOME]])
                ndrs = NDRSet(action, [ndr])
                self._ndrs[action.predicate] = ndrs

        return super().learn()

    def _create_prompt(self):
        # TODO: use ac.train_env to extract predicates, operator names, and
        # create this prompt automatically.
        """"""
        assert self._domain_name == "Glibblocks"
        # reference: https://github.com/tomsilver/pddlgym/tree/master/pddlgym/pddl/glibblocks.pddl
        prompt = """# Fill in the <TODO> to complete the PDDL domain.
        
        (define (domain glibblocks)
    (:requirements :strips :typing)
    (:types block robot)
    (:predicates 
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty ?x - robot)
        (handfull ?x - robot)
        (holding ?x - block)
        (pickup ?x - block)
        (putdown ?x - block)
        (stack ?x - block ?y - block)
        (unstack ?x - block)
    )

    ; (:actions pickup putdown stack unstack)

    (:action pick-up
        :parameters (?x - block <TODO>)
        :precondition (and
            (pickup ?x) 
            <TODO>
        )
        :effect (and
            <TODO>
        )
    )

    (:action put-down
        :parameters (?x - block <TODO>)
        :precondition (and 
            (putdown ?x)
            <TODO>
        )
        :effect (and 
            <TODO>
        )

    (:action stack
        :parameters (?x - block ?y - block <TODO>)
        :precondition (and
            (stack ?x ?y)
            <TODO>
        )
        :effect (and 
            <TODO>
        )
    )

    (:action unstack
        :parameters (?x - block <TODO>)
        :precondition (and
            (unstack ?x)
            <TODO>
        )
        :effect (and 
            <TODO>
        )
    )
)"""

        return prompt

    def _query_llm(self, prompt):
        # TODO cache and make settings
        # reference: https://github.com/Learning-and-Intelligent-Systems/llm4pddl/blob/main/llm4pddl/llm_interface.py

        # TODO: uncomment. Leaving commented for now to avoid spurious queries
        # of the expensive open AI API. Also we might want to use ChatGPT instead...
        # completion = openai.Completion.create(
        #     engine="code-davinci-002",
        #     prompt=prompt,
        #     max_tokens=500,
        #     temperature=0,
        # )
        # response = completion.choices[0].text

        # This is a response from ChatGPT (manually collected)
        response = """(define (domain glibblocks)
(:requirements :strips :typing)
(:types block robot)
(:predicates
(on ?x - block ?y - block)
(ontable ?x - block)
(clear ?x - block)
(handempty ?x - robot)
(handfull ?x - robot)
(holding ?x - block)
(pickup ?x - block)
(putdown ?x - block)
(stack ?x - block ?y - block)
(unstack ?x - block)
)

; (:actions pickup putdown stack unstack)

(:action pick-up
    :parameters (?r - robot ?x - block)
    :precondition (and
        (pickup ?x) 
        (handempty ?r)
        (clear ?x)
    )
    :effect (and
        (handfull ?r)
        (not (handempty ?r))
        (not (clear ?x))
        (holding ?x)
    )
)

(:action put-down
    :parameters (?r - robot ?x - block)
    :precondition (and 
        (putdown ?x)
        (handfull ?r)
        (holding ?x)
    )
    :effect (and 
        (handempty ?r)
        (not (handfull ?r))
        (clear ?x)
        (not (holding ?x))
    )
)

(:action stack
    :parameters (?r - robot ?x - block ?y - block)
    :precondition (and
        (stack ?x ?y)
        (handfull ?r)
        (holding ?x)
        (clear ?y)
    )
    :effect (and 
        (handempty ?r)
        (not (handfull ?r))
        (not (holding ?x))
        (not (clear ?y))
        (on ?x ?y)
    )
)

(:action unstack
    :parameters (?r - robot ?x - block)
    :precondition (and
        (unstack ?x)
        (handempty ?r)
        (on ?x ?y)
    )
    :effect (and 
        (handfull ?r)
        (not (handempty ?r))
        (clear ?x)
        (not (on ?x ?y))
        (holding ?x)
    )
)

)
"""

        return response

    def _llm_output_to_operators(self, llm_output):
        # Parse the LLM output using PDDLGym.
        
        # TODO: automatically handle this and other cases of malformed LLM output.
        # In this case, we need to add a missing parameter.
        llm_output =llm_output.replace("""(:action unstack
    :parameters (?r - robot ?x - block)""", """(:action unstack
    :parameters (?r - robot ?x - block ?y - block)""")

        domain_fname = tempfile.NamedTemporaryFile(delete=False).name
        with open(domain_fname, "w", encoding="utf-8") as f:
            f.write(llm_output)
        domain = PDDLDomainParser(domain_fname)
        return list(domain.operators.values())
