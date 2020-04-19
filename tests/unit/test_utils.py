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
episode_parameters = []
episode_parameters_ids = []
steps = [('a','b','c'), (1,2,3), (utils.Step(0, 1), utils.Step(4, 7)), (True, 'False')]
rewards = [1, 2, 7.5, 43.0984]
for step, reward in zip(steps, rewards):
    episode_parameters.append((step, reward))
    episode_parameters_ids.append('steps={}, reward={}'.format(step, reward))

@pytest.fixture(params=episode_parameters, ids=episode_parameters_ids)
def episode_params(request):
    return request.param

def test_episode_creation(episode_params):
    episode = utils.Episode(episode_params[0], episode_params[1])
    assert episode.steps == episode_params[0]
    assert episode.reward == episode_params[1]


# Test generate_batch
# creation
# negative_input
