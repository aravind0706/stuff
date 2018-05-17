import mujoco_py
from mujoco_py import MjSim, MjViewer
import numpy as np
import gym
from mujoco_py.generated import const
from gym import spaces, error
from gym.utils import seeding
import os
import numpy as np
import six

class MujocoEnv(gym.Env):
    """SuperClass
    """

    def __init__(self, model_path, frame_skip = 5, model_stat_extent=None):
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        if model_stat_extent is not None:
            self.model.stat.extent = model_stat_extent
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        #self.mjviewer = mjviewer
        self.t = 0
        self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0/self.dt))
        }
        self.action_dim = (self.model.nu,) # format needed for HER
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        self.obs_dim = (observation.size,) # format needed for HER
        #assert not done
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(observation.size)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return dict(qpos=self.data.qpos[:].copy(), qvel=self.data.qvel[:].copy())

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implemented in subclasses.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def reset(self):
        """
        Can be overridden in the subclasses.
        """
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)


    def get_img(self, width=480, height=480):
        return self.sim.render(width, height)
    

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
