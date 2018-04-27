# RLProject

## To-Do

- [x] Test on MountainCar to make sure it's working
- [ ] Test on MuJoCo - see if we can get similar results to Berkeley paper
- [x] MPC with softmax instead of hard-max
- [x] Add reward predictions
- [ ] Add value function
- [ ] Implement REINFORCE over it, see if we can get a model that's better tuned to planning
- [ ] Investigate upgrades to MPC (look at optimal control)
- [ ] Think about how do we make the planner differentiable?

## Practical Issues

- [x] Work out the proper data split in aggregation. (EDIT - turns out it's a 10-90 random data - current model split)
- [x] Maybe fix the dataset's size?
- [x] Work out issue with reward oracle/gnerating and evaluating trajectories
- [x] Reconsider how we combine D_rand and D_RL
- [x] Actions are normalised by the net, but not normalised in our sampling - need to fix.
- [x] Maintain a persistent 90-10 D_RL - D_rand split after first iteration
- [x] Parallelise the trajectory sampling (vectorized instead)
- [ ] Parallelise reward oracle?
- [ ] Add proper model-saving and trajectory-saving utilities
- [ ] Add some utilities to make it easier to evaluate the quality of the model and reward prediction
- [ ] Add functionality to save videos of trajectories (makes it easier to qualitatively assess how it's doing)


