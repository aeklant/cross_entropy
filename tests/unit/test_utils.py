import cross_entropy.utils as utils
import pytest


# Test Step
step_parameters = []
states = ['state1', 's2', 'puppies', 4, 'five']
actions = ['a1', 'action2', 3.0, 'four', 5]
step_params_ids = []
for state, action in zip(states, actions):
    step_parameters.append((state, action))
    step_params_ids.append('state={}, action={}'.format(state, action))

@pytest.fixture(params=step_parameters, ids=step_params_ids)
def step_params(request):
    return request.param

@pytest.mark.Step
@pytest.mark.creation
def test_step_creation(step_params):
    step = utils.Step(step_params[0], step_params[1])
    assert step.state == step_params[0]
    assert step.action == step_params[1]


# Test Episode
# creation
# negative_input


# Test generate_batch
# creation
# negative_input
