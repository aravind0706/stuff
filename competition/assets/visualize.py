import mujoco_py
import numpy as np
from mujoco_py import MjSim, MjViewer
import time
import sys
from scipy.misc import imsave
import cv2

xml_file = sys.argv[1]
if not xml_file:
    xml_file = 'half_cheetah.xml'
model = mujoco_py.load_model_from_path(xml_file)
sim = MjSim(model)
viewer = MjViewer(sim)

num_actions = sim.model.nu
t = 0

while t < 1000000:
    sim.data.ctrl[:] = 1.*np.random.uniform(-1., 1., size=(num_actions,))
    for _ in range(5):
        sim.step()
        #img = sim.render(width=1920, height=1080)
        #img = cv2.flip(img, 0)
        #img = cv2.flip(img, 1)
        #imsave('img_' + str(t) + '.jpg', img)
        t += 1
        viewer.render()
        #time.sleep(.1)
