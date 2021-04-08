<hr>

##### Codes for paper *Federated Multi-Agent Actor-Critic Learning for Age Sensitive Mobile Edge Computing*

- To simulate the MEC systems in the paper, standard *gym* modules are implemented by `MEC_env/mec_def.py` and `MEC_env/mec_env.py`.
- An edge-federated actor-critic RL framework with mixed policies,  abbreviated  as  EdgeFed  H-MAAC, is developed in `MAAC_agent.py`.
- A mixed DDPG based algorithm `AC_agent.py` is also implemented as a baseline.
- Run `*_run.py` to test the algorithms in the simulated MEC system.