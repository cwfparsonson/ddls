from collections import defaultdict
from sqlitedict import SqliteDict
import pickle
import gzip
import threading
import time
import copy
import numpy as np


class Logger:
    def __init__(self,
                 path_to_save: str,
                 actor_step_log_freq: int = None,
                 episode_log_freq: int = None,
                 epoch_log_freq: int = None,
                 use_sqlite_database: bool = False,
                 **kwargs):

        self.path_to_save = path_to_save

        log_freqs = [actor_step_log_freq, episode_log_freq, epoch_log_freq]
        if log_freqs.count(None) != 2:
            raise Exception(f'Must specify one, and only one, of actor_step_log_freq, episode_log_freq, and epoch_log_freq. The rest should be None.')
        self.actor_step_log_freq = actor_step_log_freq
        self.episode_log_freq = episode_log_freq
        self.epoch_log_freq = epoch_log_freq

        self.use_sqlite_database = use_sqlite_database

        self.reset()

    def reset(self):
        self.save_thread = None

    def write(self, data: dict):
        '''
        Args:
            data: Nested dict of dicts
        
        e.g. if data = {'epoch_stats': <data>, 'episode_stats': <data>}, then
        filenames will be epoch_stats.<extension>, episode_stats.<extension>.
        '''
        if self.save_thread is not None:
            self.save_thread.join()
        self.save_thread = threading.Thread(
                                            target=self._save_data, 
                                            args=(copy.deepcopy(data),)
                                            )
        self.save_thread.start()

    def _save_data(self, data: dict):
        start_time = time.time()
        for log_name, log in data.items():
            log_path = self.path_to_save + f'{log_name}'

            if self.use_sqlite_database:
                # update log sqlite database under database folder
                with SqliteDict(log_path + '.sqlite') as _log:
                    if len(_log) == 0:
                        # initialise
                        for key, val in log.items():
                            if not isinstance(val, list):
                                val = [val]
                            try:
                                _log[key] = val
                            except pickle.PicklingError:
                                pass
                    else:
                        for key, val in log.items():
                            if key not in {'config', 'experiment_id', 'trial_id', 'pid', 'hostname', 'node_ip'}:
                                if not isinstance(val, list):
                                    val = [val]
                                if key in _log:
                                    # extend vals list
                                    _log[key] += val
                                else:
                                    # create val
                                    _log[key] = val
                    _log.commit()
                    _log.close()
            else:
                # save log as pkl
                with gzip.open(log_path + '.pkl', 'wb') as f:
                    pickle.dump(log, f)

        print(f'Saved {list(data.keys())} data to {self.path_to_save} in {(time.time() - start_time):.4f} s.')
