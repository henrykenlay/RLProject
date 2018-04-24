# RLProject

## To-Do

- [x] Test on MountainCar to make sure it's working
- [ ] Test on MuJoCo - see if we can get similar results to Berkeley paper
- [x] MPC with softmax instead of hard-max
- [ ] Implement REINFORCE over it, see if we can get a model that's better tuned to planning (add rewards)
- [ ] Add reward or value predictions?

## Practical Issues

- [x] Work out the proper data split in aggregation. (EDIT - turns out it's a 10-90 random data - current model split)
- [x] Maybe fix the dataset's size?
- [x] Work out issue with reward oracle/gnerating and evaluating trajectories
- [x] Reconsider how we combine D_rand and D_RL
- [x] Actions are normalised by the net, but not normalised in our sampling - need to fix.
- [ ] Maintain a persistent 90-10 D_RL - D_rand split after first iteration
- [ ] Parallelise the trajectory sampling

