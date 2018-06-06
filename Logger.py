import pandas as pd

class Logger():
    
    def __init__(self, log_dir, experiment_name, columns = ['episodes', 'total_rewards', 'state_val', 'reward_val']):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.fname = '{}/{}.csv'.format(log_dir, experiment_name)
        self.data = []
        self.columns = columns
    
    def log(self, values):
        assert len(values) == len(self.columns)
        self.data.append(values)
        self.makefile()
        
    def makefile(self):
        data = pd.DataFrame(self.data, columns = self.columns)
        data.to_csv(self.fname, index=False)