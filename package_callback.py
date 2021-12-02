from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class DeliveredPackagesLogger(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(DeliveredPackagesLogger, self).__init__(verbose)
        #self.state = state
        #self.t = 0
        #self.packages_delivered = 0
        self.average_period = 1000 # 1000 steps
        self.mean_packages_delivered = 0

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        #value = np.random.random()
        # Gives err: self.training_env.state.packages_delivered

        #print('LOGGER STATE:')
        #self.training_env.get_attr('state')[0].render()
        packages_delivered = self.training_env.get_attr('state')[0].packages_delivered

        self.mean_packages_delivered += packages_delivered
        
        # training_env is a DummyVecEnv
        #self.logger.record('packages_delivered', packages_delivered)
        #print(type(packages_delivered))
        # if packages_delivered > 0:
        #     print('Some packages were delivered.')
        #     print('Packages delivered:', packages_delivered)
            
        #     value = 5000.0
        # else:
        #     #value = np.random.random()
        #     value = 5.0
        #self.logger.record('random_value', value)
        # https://github.com/DLR-RM/stable-baselines3/issues/506
        
        if self.num_timesteps % self.average_period == 0: 
            self.logger.record('mean_packages_delivered', self.mean_packages_delivered / self.average_period)
            self.mean_packages_delivered = 0
            self.logger.dump(self.num_timesteps)
        return True
