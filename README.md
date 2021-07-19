## EdgeFed H-MAAC: Edge Federated Heterogeneous Multi-agent Actor-Critic 

This repository contains a *gym* module for UAV-assisted MEC environment simulation and a TensorFlow implementation of EdgeFed H-MAAC framework.

### Run


- To simulate the MEC systems in the paper, standard *gym* modules are implemented by `MEC_env/mec_def.py` and `MEC_env/mec_env.py`.
- An edge-federated actor-critic RL framework with mixed policies,  abbreviated  as  EdgeFed  H-MAAC, is developed in `MAAC_agent.py`.
- A mixed DDPG based algorithm `AC_agent.py` is also implemented as a baseline.
- Run `*_run.py` to test the algorithms in the simulated MEC system.


### References

If you find the codes useful, please cite the following papers:
* *Federated Multi-Agent Actor-Critic Learning for Age Sensitive Mobile Edge Computing* [J]. IEEE Internet of Things Journal, 2021.

* *An Edge Federated MARL Approach for Timeliness Maintenance in MEC Collaboration* [C]//2021 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2021: 1-6.

<hr>
