from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.env.randomize_attributes = True
        self.sim.train_val_scenario = 'all_scenarios'
        self.env.val_size = 50

class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'lstm_rl'


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.train.target_update_interval = 500
        self.train.evaluation_interval = 500
        self.train.checkpoint_interval = 500
