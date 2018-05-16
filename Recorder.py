import os
from PIL import Image
import subprocess
import shutil

class Recorder():
    
    def __init__(self, experiment_name, count):
        self.experiment_name = experiment_name
        self.count = count
        os.makedirs('frames', exist_ok=True)
        os.makedirs('frames/{}'.format(self.experiment_name), exist_ok=True)
        os.makedirs(os.path.join('frames', '{}-{}'.format(self.experiment_name, self.count)))
        
    def record_frame(self, image_data, timestep):
        img = Image.fromarray(image_data, 'RGB')
        fname = os.path.join('frames', '{}-{}'.format(self.experiment_name, self.count), 'frame-%.10d.png' % timestep)
        img.save(fname)
        
    def make_movie(self):
        frames = 'frames/{}-{}/frame-%010d.png'.format(self.experiment_name, self.count)
        movie = 'frames/{}/{}.mp4'.format(self.experiment_name, self.count)
        string = "ffmpeg -framerate 24 -y -i {} -r 30 -pix_fmt yuv420p {}".format(frames, movie)
        subprocess.call(string.split())
        shutil.rmtree('frames/{}-{}'.format(self.experiment_name, self.count))