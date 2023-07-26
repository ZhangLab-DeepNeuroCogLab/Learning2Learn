from avalanche.training.plugins import StrategyPlugin


class CustomTrainingPlugin(StrategyPlugin):
    def __init__(self, epochs_per_task):
        super().__init__()
        self.epochs_per_task = epochs_per_task

    def after_training_iteration(self, strategy, **kwargs):
        if strategy.current_iteration % strategy.task_length == 0:
            # Update the number of epochs per task
            strategy.train_epochs = self.epochs_per_task
 