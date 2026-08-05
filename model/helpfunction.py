import os
import cv2
import base64
from PIL import Image
import torch
from torchvision.utils import make_grid
import logging
from os import makedirs


from IPython.display import display, HTML, clear_output

def to_device(data, device):

    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader:


    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):

        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):

        return len(self.dl)


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)



def get(element: torch.Tensor, t: torch.Tensor):



    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)


def setup_log_directory(config):


    if os.path.isdir(config.root_log_dir):

        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(config.root_log_dir)]


        last_version_number = max(folder_numbers)


        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir


    log_dir        = os.path.join(config.root_log_dir,        version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)


    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir

def frames2vid(images, save_path):

    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]



    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))


    for image in images:
        video.write(image)



    video.release()
    return

def display_gif(gif_path):
    b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))


def logger_configuration(filename, phase, save_log=True):
    logger = logging.getLogger(filename)
    workdir = '/home/cxq/code_10.23/history/{}'.format(filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    makedirs(workdir)
    makedirs(samples)
    makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger

class AverageMeter:


    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
