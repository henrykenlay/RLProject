# RLProject

## To-Do

- [ ] Test on MountainCar to make sure it's working
- [ ] Test on MuJoCo - see if we can get similar results to Berkeley paper
- [ ] MPC with softmax instead of hard-max
- [ ] Implement REINFORCE over it, see if we can get a model that's better tuned to planning
- [ ] Add reward or value predictions?

## Practical Issues

- [x] Work out the proper data split in aggregation. (EDIT - turns out it's a 10-90 random data - current model split)
- [x] Maybe fix the dataset's size?
- [ ] Parallelise the trajectory sampling

