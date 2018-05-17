from predprey.ppo.common import tf_util as U
import gym, logging
from predprey.ppo.common import logger
from predprey.ppo import mlp_policy, pposgd_simple
from predprey.envs.pred_prey import XYZ # To be changed

def train(env_name, num_timesteps, horizon, threshold, timesteps_per_actorbatch, optim_batchsize, optim_epochs, optim_stepsize, gamma, config,  dump_directory, render, **kwargs):
    U.make_session(num_cpu=1).__enter__()
    if env_name == 'reacher-obstacles':
        env = ReacherObstacleEnv(threshold=threshold, option='not-sparse', xml_file='mujoco_models/reacher_obstacles/config1.xml')
    elif env_name == 'pointmass-obstacles':
        env = PointMassObstacleEnv(threshold=threshold, option='not-sparse')
    elif env_name == 'ant-obstacles-dense':
        env = AntObstacleDenseEnv(config=config)
    elif env_name == 'ant-obstacles-dense-harder':
        env = AntObstacleDenseHarderEnv(config=config)
    elif env_name == 'ant-obstacles-dense-hardest':
        env = AntObstacleDenseHardestEnv(config=config)
    elif env_name == 'ant-obstacles-dense-deadlier':
        env = AntObstacleDenseDeadlierEnv(config=config)
    elif env_name == 'ant-obstacles-dense-deadliest':
        env = AntObstacleDenseDeadliestEnv(config=config)
    elif env_name == 'pusher':
        env = PusherVisionEnv(config=1, option='non-sparse')
    elif env_name == 'striker':
        env = StrikerEnv()
    elif env_name == 'humanoid-obstacles-dense':
        env = HumanoidObstacleDenseEnv(config=config)

    def policy_fn(name, ob_dim, ac_dim):
        return mlp_policy.MlpPolicy(name=name, ob_dim=ob_dim, ac_dim=ac_dim,
            hid_size=128, num_hid_layers=2)

    
    pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                horizon=horizon,
                timesteps_per_actorbatch=timesteps_per_actorbatch,
                optim_batchsize=optim_batchsize,
                optim_epochs=optim_epochs,
                optim_stepsize=optim_stepsize,
                gamma=gamma,
                dump_directory=dump_directory,
                render=render,
                clip_param=0.2, entcoeff=0.0,
                #optim_stepsize=5e-5, 
                lam=0.95, schedule='constant',
    )

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-name', help='environment', default='reacher-obstacles')
    parser.add_argument('--num-timesteps', type=int, default=int(100e6))
    parser.add_argument('--threshold', type=float, default=0.075)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--timesteps-per-actorbatch', type=int, default=4096)
    parser.add_argument('--optim-batchsize', type=int, default=256)
    parser.add_argument('--optim-epochs', type=int, default=1)
    parser.add_argument('--optim-stepsize', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--config', type=int, default=11)
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()
    logdir = 'ppo-constant-schedule-' + args.env_name + '-config-' + str(args.config) + '-num-timesteps-' + str(args.num_timesteps) + '-horizon-' + str(args.horizon) + '-batchsize-' + str(args.timesteps_per_actorbatch) + '-gamma-' + str(args.gamma) + '-optim-batchsize-' + str(args.optim_batchsize) + '-optim-epochs-' + str(args.optim_epochs) + '-optim-stepsize-' + str(args.optim_stepsize)
    logdir = '/Users/aravind/' + logdir
    #logdir = '/tmp/latentplan/ppo/' + logdir
    dumpdir = logdir
    logger.configure(logdir)
    train(args.env_name, 
            num_timesteps=args.num_timesteps, 
            horizon=args.horizon, 
            threshold=args.threshold, 
            timesteps_per_actorbatch=args.timesteps_per_actorbatch,
            optim_batchsize=args.optim_batchsize,
            optim_epochs=args.optim_epochs,
            optim_stepsize=args.optim_stepsize,
            gamma=args.gamma,
            config=args.config,
            dump_directory=dumpdir, 
            render=args.render)


if __name__ == '__main__':
    main()
