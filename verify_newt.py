"""Post-install validation for the Newt / MMBench conda environment.

Level 1: imports + versions for every pinned dependency.
Level 2: functional witness -- instantiate and step real environments from
         each task family, and load the repo's task registry.
Runs on a CPU node; GPU checks are reported but not required.
"""
import importlib
import json
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")  # conda glew/mesalib absent -> EGL is the working backend
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

results = {"imports": {}, "witness": {}, "gpu": {}}

PKGS = [
    "torch", "torchvision", "torchrl", "tensordict", "numpy", "transformers",
    "kornia", "mujoco", "dm_control", "gymnasium", "metaworld", "mani_skill",
    "ogbench", "robodesk", "hydra", "omegaconf", "submitit", "h5py", "imageio",
    "moviepy", "wandb", "ale_py", "termcolor", "tqdm", "glfw",
]

for name in PKGS:
    try:
        m = importlib.import_module(name)
        results["imports"][name] = getattr(m, "__version__", "ok")
    except Exception as e:
        results["imports"][name] = f"FAIL: {type(e).__name__}: {e}"

# ---- GPU visibility (informational on a CPU node) ----
try:
    import torch
    results["gpu"] = {
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
except Exception as e:
    results["gpu"] = {"FAIL": repr(e)}


def witness(label, fn):
    try:
        results["witness"][label] = fn()
    except Exception:
        results["witness"][label] = "FAIL: " + traceback.format_exc(limit=3).strip().splitlines()[-1]


# ---- DMControl (state) ----
def _dmc():
    from dm_control import suite
    env = suite.load("walker", "walk")
    ts = env.reset()
    import numpy as np
    a = np.zeros(env.action_spec().shape)
    ts = env.step(a)
    return {"obs_keys": sorted(ts.observation.keys()), "reward": float(ts.reward)}


# ---- MuJoCo raw ----
def _mujoco():
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint type='free'/>"
                                       "<geom size='.1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    return {"nq": int(m.nq), "time": float(d.time)}


# ---- Meta-World (Hansen fork) ----
def _metaworld():
    import metaworld
    benches = [x for x in dir(metaworld) if not x.startswith("_")]
    return {"attrs": benches[:12]}


# ---- ManiSkill ----
def _maniskill():
    import mani_skill
    from mani_skill.utils.registration import REGISTERED_ENVS
    asset_dir = os.environ.get("MANISKILL_ASSET_DIR",
                               os.path.expanduser("~/.maniskill"))
    return {
        "version": mani_skill.__version__,
        "n_registered_envs": len(REGISTERED_ENVS),
        "asset_dir": asset_dir,
        "asset_dir_exists": os.path.isdir(asset_dir),
    }


# ---- OGBench ----
def _ogbench():
    import ogbench
    return {"version": getattr(ogbench, "__version__", "ok")}


# ---- Repo task registry + config ----
def _repo():
    tj = json.load(open(os.path.expanduser("~/newt/tasks.json")))
    if isinstance(tj, dict):
        summary = {k: (len(v) if hasattr(v, "__len__") else v) for k, v in tj.items()}
    else:
        summary = {"n_tasks": len(tj)}
    return summary


def _cfg():
    # config.py defines a hydra *structured* config (dataclass), not a YAML.
    sys.path.insert(0, os.path.expanduser("~/newt/tdmpc2"))
    from omegaconf import OmegaConf
    import config as newt_config
    cfg = OmegaConf.structured(newt_config.Config)
    keys = list(cfg.keys())
    return {"n_config_keys": len(keys),
            "model_size": cfg.get("model_size"),
            "task": cfg.get("task"),
            "obs": cfg.get("obs"),
            "steps": cfg.get("steps"),
            "batch_size": cfg.get("batch_size"),
            "some_keys": keys[:20]}


def _make_env():
    # Full repo integration: build a single-task env through the repo's own
    # make_env() factory, then step it.
    sys.path.insert(0, os.path.expanduser("~/newt/tdmpc2"))
    from omegaconf import OmegaConf
    import config as newt_config
    from envs import make_env
    cfg = OmegaConf.structured(newt_config.Config)
    cfg.task = "walker-walk"
    cfg.obs = "state"
    cfg.multitask = False
    env = make_env(cfg)
    obs = env.reset()
    import numpy as np
    act = env.rand_act() if hasattr(env, "rand_act") else \
        np.zeros(env.action_space.shape, dtype=np.float32)
    step = env.step(act)
    return {"task": "walker-walk",
            "obs_shape": tuple(np.asarray(obs).shape),
            "action_dim": int(np.prod(env.action_space.shape)),
            "step_len": len(step)}


witness("mujoco", _mujoco)
witness("dm_control", _dmc)
witness("metaworld", _metaworld)
witness("maniskill", _maniskill)
witness("ogbench", _ogbench)
witness("repo_tasks", _repo)
witness("hydra_config", _cfg)
def _render():
    # obs=rgb and video logging both need a working offscreen GL context.
    from dm_control import suite
    import numpy as np
    env = suite.load("walker", "walk")
    env.reset()
    px = env.physics.render(height=64, width=64, camera_id=0)
    return {"backend": os.environ["MUJOCO_GL"], "frame_shape": tuple(px.shape),
            "dtype": str(px.dtype), "nonzero_frac": round(float((px > 0).mean()), 4)}


witness("repo_make_env", _make_env)
witness("egl_offscreen_render", _render)

print(json.dumps(results, indent=2, sort_keys=True))
with open("verify_newt.json", "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)

fails = [k for k, v in results["imports"].items() if str(v).startswith("FAIL")]
wfails = [k for k, v in results["witness"].items() if str(v).startswith("FAIL")]
print(f"\nIMPORT_FAILURES={fails}")
print(f"WITNESS_FAILURES={wfails}")
