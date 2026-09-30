import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['MS_SKIP_ASSET_DOWNLOAD_PROMPT'] = '1'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings; warnings.filterwarnings('ignore')
import copy, json, traceback
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from config import Config, parse_cfg
from envs import make_env

cs = ConfigStore.instance()
cs.store(name='config', node=Config)

# one representative task per MMBench domain
TASKS = ['walker-walk', 'mw-reach', 'ms-pick-cube', 'mujoco-hopper',
         'og-point-maze', 'rd-push-red', 'pygame-pong', 'atari-pong',
         'bipedal-walker-flat']


def _num(x):
    try:
        return float(np.asarray(x, dtype=np.float64).mean())
    except Exception:
        return None


@hydra.main(version_base=None, config_name='config')
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    out = {}
    for t in TASKS:
        try:
            c2 = copy.deepcopy(cfg)
            c2.task = t
            c2.num_envs = 1
            c2.env_mode = 'sync'
            c2.obs = 'state'
            c2 = parse_cfg(c2)
            env = make_env(c2)
            obs = env.reset()
            act = env.rand_act()
            step = env.step(act)
            rew = step[1] if isinstance(step, (tuple, list)) else step
            out[t] = {'obs_shape': list(np.asarray(obs).shape),
                      'action_shape': list(np.asarray(act).shape),
                      'reward': _num(rew)}
            try:
                env.close()
            except Exception:
                pass
        except Exception:
            tb = traceback.format_exc().strip().splitlines()
            out[t] = 'FAIL: ' + tb[-1]
        print(t, '->', out[t], flush=True)
    json.dump(out, open(os.path.expanduser('~/newt/smoke_env.json'), 'w'), indent=2)
    print('FAILS=' + str([k for k, v in out.items() if str(v).startswith('FAIL')]))


if __name__ == '__main__':
    main()
