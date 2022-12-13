try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import TensorBoardOutputFormat
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    class SummaryWriterCallback(object):
        """
        Do not use, this class is a template when stable baselines3 is not installed.
        
        It represents `stable_baselines3.common.callbacks import BaseCallback` 
        and `from stable_baselines3.common.logger import TensorBoardOutputFormat`
        """

class SummaryWriterCallback(BaseCallback):

    def __init__(
        self,
        save_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_training_start(self):
        # self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            n_save = self.n_calls // self.save_freq
            # self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
            # self.tb_formatter.writer.flush()
            for name, weight in self.model.policy.named_parameters():
                self.tb_formatter.writer.add_histogram(name, weight, n_save)
                # self.tb_formatter.add_histogram(f'{name}.grad',weight.grad, epoch)