from collections import namedtuple
import torch


Step = namedtuple('Step', ['state', 'action'])
Episode = namedtuple('Episode', ['steps', 'reward'])


@torch.no_grad()
def generate_batch(model, action_selector, env, batch_size):
    """
    Generates a batch of episodes from an environment

    Parameters
    ----------
    model: Callable
        Transforms an input state into an output
    action_selector: Callable
        Takes the output from model and picks an action
    env: gym.Env instance
        An environment from the gym package. Will take an action and create a new
        input observation for the model
    batch_size: int
        The number of episodes to return as a batch

    Yields
    ------
    batch: tuple
        A collection of episodes of size batch_size
    """
    batch = []

    while True:
        obs = env.reset()
        is_done = False
        total_reward = 0.0
        episode = Episode(steps=[], reward=total_reward)

        while not is_done:
            action = action_selector(model([obs]))

            episode.steps.append(Step(state=obs, action=action))
            obs, reward, is_done, info = env.step(action)
            total_reward += reward
            
        episode.reward = total_reward
        batch.append(episode)

        if len(batch) == batch_size:
            yield batch
            batch = []
