# A neural architecture for approximating flows of dynamical systems with inputs

See our paper [Miguel Aguiar, Amritam Das and Karl H. Johansson, _Learning Flow Functions from Data with Applications to Nonlinear Oscillators_ (2023)](https://www.sciencedirect.com/science/article/pii/S240589632302147X) for a description of the method.

The requirements for running the code are PyTorch and NumPy.

To run e.g. the Van der Pol example, first create a data file as follows:
```shell
  mkdir -p outputs
  python experiments/vdp_flow.py --control_delta 0.2 --time_horizon 15 \
     --n_trajectories 200 --n_samples 200 vdp_standard_data.pth
```
This will create a data file in `./outputs/vdp_standard_data.pth`.
To train a model on this data, run
```shell
  python experiments/load_data.py  --lr=0.0005 --n_epochs=200 --batch_size=512 \
    --es_delta=1e-4 --es_patience=15 --sched_patience=5 --sched_factor=10 \
    --control_rnn_size=16 --encoder_depth=3 --encoder_size=1 --decoder_depth=3 --decoder_size=1 \
    --experiment_id=vdp_standard  outputs/vdp_standard_data.pth
```
which will create a .pth file with the model parameters in the directory `./outputs/vdp_standard`.
To simulate the trained model, you can use the help script `experiments/scripts/interactive_test.py`.
