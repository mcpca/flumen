import torch

torch.set_default_dtype(torch.float32)

from torch.utils.data import DataLoader, random_split
from trajectory import (TrajectoryGenerator, TrajectoryDataset,
                        SequenceGenerator)
from flow_model import CausalFlowModel
from train import EarlyStopping, train, validate
from dynamics import Dynamics

from meta import Meta, instantiate_model

import time


def simulate(dynamics: Dynamics, control_generator: SequenceGenerator,
             control_delta, n_trajectories, n_samples, time_horizon,
             examples_per_traj):
    trajectory_generator = TrajectoryGenerator(
        dynamics,
        control_delta=control_delta,
        control_generator=control_generator)

    traj_data = TrajectoryDataset(trajectory_generator,
                                  n_trajectories=n_trajectories,
                                  n_samples=n_samples,
                                  time_horizon=time_horizon,
                                  examples_per_traj=examples_per_traj)

    return traj_data, trajectory_generator


def preprocess(traj_data, batch_size, split):
    norm_center, norm_weight = traj_data.whiten_targets()

    train_len = int((float(split) / 100.) * len(traj_data))
    train_data, val_data = random_split(traj_data,
                                        lengths=(train_len,
                                                 len(traj_data) - train_len))

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, norm_center, norm_weight


def training_loop(meta, model, loss_fn, optimizer, sched, early_stop, train_dl,
                  val_dl, device, max_epochs):
    print('Epoch :: Loss (Train) :: Loss (Val) :: Best (Val)')
    print('=================================================')

    start = time.time()

    for epoch in range(max_epochs):
        loss = 0.

        model.train()
        for example in train_dl:
            loss += train(example, loss_fn, model, optimizer, device)

        loss /= len(train_dl)

        model.eval()
        val_loss = validate(val_dl, loss_fn, model, device)
        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {loss:>7e} :: {val_loss:>7e} :: {early_stop.best_val_loss:>7e}"
        )

        if early_stop.best_model:
            meta.save_model(model)

        meta.register_progress(loss, val_loss, early_stop.best_model)

        if early_stop.early_stop:
            break

    train_time = time.time() - start

    # Compute train set loss for the optimal model.
    model = meta.load_model()
    model.to(device)
    train_loss = validate(train_dl, loss_fn, model, device)
    meta.train_loss_best = train_loss

    meta.save(train_time)

    return train_time


def sim_and_train(args,
                  dynamics: Dynamics = None,
                  control_generator: SequenceGenerator = None,
                  load_data=False):

    if load_data:
        traj_data, traj_generator = torch.load(args.load_data)
        dynamics = traj_generator._dyn

    else:
        examples_per_traj = (args.n_samples if args.generate_test_set else
                             args.examples_per_traj)

        traj_data, traj_generator = simulate(
            dynamics,
            control_generator,
            control_delta=args.control_delta,
            n_trajectories=args.n_trajectories,
            n_samples=args.n_samples,
            time_horizon=args.time_horizon,
            examples_per_traj=examples_per_traj)

    if args.save_data:
        torch.save((traj_data, traj_generator), f'outputs/{args.save_data}')

    if args.generate_test_set:
        torch.save(traj_data, f'outputs/{args.generate_test_set}')
        return

    train_dl, val_dl, norm_center, norm_weight = preprocess(
        traj_data, batch_size=args.batch_size, split=args.train_val_split)

    model: CausalFlowModel = instantiate_model(args, dynamics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    meta = Meta(args,
                traj_generator,
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
                               device,
                               max_epochs=args.n_epochs)

    print(f"Training took {train_time:.2f} seconds.")
