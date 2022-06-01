import gym
from gym.envs.registration import registry, make, spec
from .humanoid_swimmer_env import HumanoidSwimmerEnv


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id="HumanoidSwimmerEnv-v1",
    entry_point="my_pybullet_envs:HumanoidSwimmerEnv",
    max_episode_steps=240,
)

def getList():
    btenvs = [
        "- " + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find("Bullet") >= 0
    ]
    return btenvs
