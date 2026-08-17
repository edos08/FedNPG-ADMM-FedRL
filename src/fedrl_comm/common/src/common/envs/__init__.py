"""
Environment wrappers around gymnasium.make() for the MuJoCo tasks used in
the paper (Swimmer-v4, Hopper-v4, Humanoid-v4 — see Table 3) plus whatever
small-state-space env is picked for task (iii).

TODO: implement make_env(name: str, seed: int, **kwargs) -> gymnasium.Env
"""
