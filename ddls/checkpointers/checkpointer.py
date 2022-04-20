import pathlib

class Checkpointer:
    def __init__(self,
                 path_to_save: str,
                 epoch_checkpoint_freq: int = 1):
        # init checkpoints dir
        if path_to_save[-1] != '/':
            self.path_to_save = path_to_save + '/checkpoints/'
        else:
            self.path_to_save = path_to_save + 'checkpoints/'
        pathlib.Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

        self.epoch_checkpoint_freq = epoch_checkpoint_freq

    def write(self, epoch_loop):
        epoch_loop.save_agent_checkpoint(self.path_to_save)
