from gym.envs.registration import register

register(
    id='delivery-v0',
    entry_point='delivery_gym.envs:DeliveryEnv',
)