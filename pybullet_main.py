import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0.0
    steps = 0
    
    A = [0.0, 0.0]
    B = [0.0557, 0.0944] # TODO: plug in your desired points here, relative to 256x256 image
    z = 0.5
    while True:
        t = steps * 0.01
        alpha = 0.5 * (1 - math.cos(t))
        
        x = (1 - alpha) * A[0] + alpha * B[0]
        y = (1 - alpha) * A[1] + alpha * B[1]
        
        target_pos = [x, y, 0.2]
        
        print(target_pos)
        
        robot.move_ee([x, y, 0.2, 0, math.pi, math.pi], 'end')
        p.stepSimulation()
        steps += 1
        time.sleep(0.01)

if __name__ == '__main__':
    user_control_demo()
