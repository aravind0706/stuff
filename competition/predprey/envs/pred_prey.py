import numpy as np
import time
from gym import utils
from predprey.envs.mujoco_env import MujocoEnv

class AntPredPrey(MujocoEnv):
    def __init__(self, frame_skip=5):
        self.frame_skip = frame_skip
        MujocoEnv.__init__(self, 'assets/pred_prey.xml', frame_skip)

    def step(self, a):
        # let's assume that each agent's action is concatenated together in a single "a" vector
        self.do_simulation(a, self.frame_skip) # set the sim forward by these actions

        # what's needed now?
        # you need to compute the reward, the next state for each ant, done (if any), and any info you want to return
        # for this environment, single reward doesn't make sense, so just return 0 for rwd, and in a dict, put in predator and prey reward separately
        
        # ant 0 is red, ant 1 is blue, let's assume that red is predator and blue is prey. 
        pred_torso = self.get_body_com('agent0/torso')
        prey_torso = self.get_body_com('agent1/torso')

        # Use Jason Peng's trick to exponentiate the rewards. 
        neg_distance = -np.linalg.norm(pred_torso - prey_torso)
        exp_neg_distance = np.exp(neg_distance)
        pred_rwd = exp_neg_distance
        prey_rwd = -1.*pred_rwd
        

        # what about the ants falling down? what kind of rewards should be assigned? don't terminate the episodes. why? because, if one ant fell but the other ant could still learn 
        #by trying to either go close, or go far away. 
        # so the best way to reward is to give -1 once the ant falls, but don't terminate the episode. if both fall down? ya, then maybe truncate the episode, 
        #because nothing is interesting / worth learning there. 
        
        done = False # default

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()

        pred_state = np.concatenate([qpos[:15], qvel[:15]])
        prey_state = np.concatenate([qpos[15:], qvel[15:]])

        pred_torso_z = qpos[2]
        prey_torso_z = qpos[17]

        pred_stable = np.isfinite(pred_state).all() and pred_torso_z >= .2 and pred_torso_z <= 1.
        prey_stable = np.isfinite(prey_state).all() and prey_torso_z >= .2 and prey_torso_z <= 1.

        if (not pred_stable) and prey_stable:
            pred_rwd = -1. 

        elif (not prey_stable) and pred_stable:
            prey_rwd = -5. # can't use -1 because it's in the range of usual rewards, need a stricter penalty. 

        elif (not pred_stable) and (not prey_stable):
            pred_rwd = -1.
            prey_rwd = -5.
            done = True

        rwd = 0. # dummy return to fit gym format
        ob = self._get_obs()
        return ob, rwd, done, dict(
                reward_predator=pred_rwd,
                reward_prey=prey_rwd)
        
    def _get_obs(self):
        # basically get all joints and contacts for each ant. 
        # for a start, let's assume both the ants know everything about the other. 
        # so one single feature vector serves the purpose.
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1., 1.).flat,
        ])
        

    def reset_model(self):
        # basically reset the torsos and a favorable init_qpos for each 
        qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-1., high=.1)
        qvel = self.init_qvel + np.random.randn(self.model.nv)*.1

        dist = 10
        while (dist > 4) or (dist < 1):
            pred_x_y = np.random.uniform(-3., 3., size=(2,))
            prey_x_y = np.random.uniform(-3., 3., size=(2,))
            dist = np.linalg.norm(pred_x_y - prey_x_y)
        print("dist", dist)
        qpos[:2] = pred_x_y
        qpos[15:17] = prey_x_y
        qpos[2:15] = np.array([0.55 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0, 1.0 ,0.0 ,-1.0 ,0.0 ,-1.0 ,0.0 ,1.0])
        qpos[17:] = np.array([0.55 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0, 1.0 ,0.0 ,-1.0 ,0.0 ,-1.0 ,0.0 ,1.0])
        #qpos = qpos + 
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == '__main__':
    env = AntPredPrey()
    ob = env.reset()
    t = 0
    for t in range(10000):
        action = np.random.uniform(-1., 1., size=(env.sim.model.nu,))
        ob, rwd, done, info = env.step(action)
        env.render()

