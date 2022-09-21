import torch

torch.set_default_dtype(torch.float32)

from torch.utils.data import DataLoader
from trajectory import (TrajectoryGenerator, TrajectoryDataset,
                        SequenceGenerator)
from flow_model import CausalFlowModel
from train import EarlyStopping, train, validate
from dynamics import Dynamics

from meta import Meta, instantiate_model

import time

import numpy as np
from scipy.linalg import sqrtm, inv


def simulate(dynamics: Dynamics, control_generator: SequenceGenerator,
             control_delta, n_trajectories, n_samples, time_horizon,
             examples_per_traj, split):
    if split[0] + split[1] >= 100:
        raise Exception("Invalid data split.")

    trajectory_generator = TrajectoryGenerator(
        dynamics,
        control_delta=control_delta,
        control_generator=control_generator)

    n_val_t = int(n_trajectories * (split[0] / 100.))
    n_test_t = int(n_trajectories * (split[1] / 100.))
    n_train_t = n_trajectories - n_val_t - n_test_t
    n_trajectories = (n_train_t, n_val_t, n_test_t)

    return (TrajectoryDataset(trajectory_generator,
                              n_trajectories=n,
                              n_samples=n_samples,
                              time_horizon=time_horizon,
                              examples_per_traj=examples_per_traj)
            for n in n_trajectories)


def whiten_targets(data: TrajectoryDataset, mean=None, std=None):
    if mean is None:
        mean = data.state.mean(axis=0)

    if std is None:
        std = sqrtm(np.cov(data.state.T))

    print(mean, std)

    istd = inv(std)

    data.state[:] = ((data.state - mean) @ istd).type(
        torch.get_default_dtype())

    data.init_state[:] = ((data.init_state - mean) @ istd).type(
        torch.get_default_dtype())

    return mean, std


def preprocess(train_data, val_data, test_data, batch_size, split):
    norm_center, norm_weight = whiten_targets(train_data)
    whiten_targets(val_data, norm_center, norm_weight)
    whiten_targets(test_data, norm_center, norm_weight)

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, norm_center, norm_weight


def training_loop(meta, model, loss_fn, optimizer, sched, early_stop, train_dl,
                  val_dl, test_dl, device, max_epochs):
    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: " \
            f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    start = time.time()

    for epoch in range(max_epochs):
        model.train()
        for example in train_dl:
            train(example, loss_fn, model, optimizer, device)

        model.eval()
        train_loss = validate(train_dl, loss_fn, model, device)
        val_loss = validate(val_dl, loss_fn, model, device)
        test_loss = validate(test_dl, loss_fn, model, device)

        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        if early_stop.best_model:
            meta.save_model(model)

        meta.register_progress(train_loss, val_loss, test_loss,
                               early_stop.best_model)

        if early_stop.early_stop:
            break

    train_time = time.time() - start
    meta.save(train_time)

    return train_time


def sim_and_train(args,
                  dynamics: Dynamics = None,
                  control_generator: SequenceGenerator = None,
                  load_data=False):

    if load_data:
        train_data, val_data, test_data = torch.load(args.load_data)

    else:
        examples_per_traj = (args.n_samples if args.generate_test_set else
                             args.examples_per_traj)

        train_data, val_data, test_data = simulate(
            dynamics,
            control_generator,
            control_delta=args.control_delta,
            n_trajectories=args.n_trajectories,
            n_samples=args.n_samples,
            time_horizon=args.time_horizon,
            examples_per_traj=examples_per_traj,
            split=args.data_split)

    if args.save_data and not args.generate_test_set:
        torch.save((train_data, val_data, test_data),
                   f'outputs/{args.save_data}')

    if args.generate_test_set:
        torch.save((train_data, val_data, test_data),
                   f'outputs/{args.generate_test_set}')
        return

    train_dl, val_dl, test_dl, norm_center, norm_weight = preprocess(
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
        split=args.data_split)

    model: CausalFlowModel = instantiate_model(args, train_data.state_dim,
                                               train_data.control_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    meta = Meta(args,
                train_data,
                train_data_mean=norm_center,
                train_data_std=norm_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.sched_patience,
        cooldown=args.sched_cooldown,
        factor=1. / args.sched_factor)

    mse_loss = torch.nn.MSELoss().to(device)

    early_stop = EarlyStopping(es_patience=args.es_patience,
                               es_delta=args.es_delta)

    train_time = training_loop(meta,
                               model,
                               mse_loss,
                               optimizer,
                               sched,
                               early_stop,
                               train_dl,
                               val_dl,
                               test_dl,
                               device,
                               max_epochs=args.n_epochs)

    print(f"Training took {train_time:.2f} seconds.")
