## EdgeFed H-MAAC: Edge Federated Heterogeneous Multi-agent Actor-Critic 

This repository contains a *gym* module for UAV-assisted MEC environment simulation and a TensorFlow implementation of ``EdgeFed H-MAAC`` framework.

Zhu Z, Wan S, Fan P, et al. [Federated Multiagent Actor–Critic Learning for Age Sensitive Mobile-Edge Computing](https://ieeexplore.ieee.org/abstract/document/9426913)[J]. IEEE Internet of Things Journal, 2021, 9(2): 1053-1067.

Zhu Z, Wan S, Fan P, et al. [An Edge Federated MARL Approach for Timeliness Maintenance in MEC Collaboration](https://ieeexplore.ieee.org/abstract/document/9473729)[C]//2021 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2021: 1-6.

### Run


- To simulate the MEC systems in the paper, standard *gym* modules are implemented by `MEC_env/mec_def.py` and `MEC_env/mec_env.py`.
- An edge-federated actor-critic RL framework with mixed policies,  abbreviated  as  EdgeFed  H-MAAC, is developed in `MAAC_agent.py`.
- A mixed DDPG based algorithm `AC_agent.py` is also implemented as a baseline.
- Run `*_run.py` to test the algorithms in the simulated MEC system.


### References

- If you find the codes useful, please cite the following in your manuscript:

```
@article{zhu2021federated,
  title={Federated Multiagent Actor--Critic Learning for Age Sensitive Mobile-Edge Computing},
  author={Zhu, Zheqi and Wan, Shuo and Fan, Pingyi and Letaief, Khaled B},
  journal={IEEE Internet of Things Journal},
  volume={9},
  number={2},
  pages={1053--1067},
  year={2021},
  publisher={IEEE}
}

@inproceedings{zhu2021edge,
  title={An Edge Federated MARL Approach for Timeliness Maintenance in MEC Collaboration},
  author={Zhu, Zheqi and Wan, Shuo and Fan, Pingyi and Letaief, Khaled B},
  booktitle={2021 IEEE International Conference on Communications Workshops (ICC Workshops)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

<hr>

