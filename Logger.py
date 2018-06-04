import pandas as pd

class Logger():
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.fname = '{}/{}.csv'.format(log_dir, experiment_name)
        self.data = []
    
    def log(self, timestep, rewards, val_loss0, val_loss1):
        self.data.append([timestep, rewards, val_loss0, val_loss1])
        self.makefile()
        
    def makefile(self):
        data = pd.DataFrame(self.data, columns = ['episodes', 'total_rewards', 'state_val', 'reward_val'])
        data.to_csv(self.fname, index=False)