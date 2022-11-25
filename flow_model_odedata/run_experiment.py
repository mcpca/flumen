import torch

torch.set_default_dtype(torch.float32)

from flow_model import CausalFlowModel, train, validate
from flow_model.train import EarlyStopping

from .trajectory_generator import SequenceGenerator
from .ode_experiment import ODEExperiment, instantiate_model
from .data import whiten_targets, TrajectoryDataGenerator

import time


def training_loop(experiment, model, loss_fn, optimizer, sched, early_stop,
                  train_dl, val_dl, test_dl, device, max_epochs):
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
            experiment.save_model(model)

        experiment.register_progress(train_loss, val_loss, test_loss,
                                     early_stop.best_model)

        if early_stop.early_stop:
            break

    train_time = time.time() - start
    experiment.save(train_time)

    return train_time


def run_experiment(args,
                   dynamics=None,
                   control_generator: SequenceGenerator = None,
                   load_data=False):

    if load_data:
        data = torch.load(args.load_data)
        data_generator: TrajectoryDataGenerator = data.generator

    else:
        data_generator = TrajectoryDataGenerator(
            dynamics,
            control_generator,
            control_delta=args.control_delta,
            noise_std=args.noise_std,
            n_trajectories=args.n_trajectories,
            n_samples=args.n_samples,
            time_horizon=args.time_horizon,
            split=args.data_split)

        data = data_generator.generate()

    if args.save_data:
        torch.save(data, f'outputs/{args.save_data}')

    train_mean, train_std, train_istd = whiten_targets(data)

    experiment = ODEExperiment(args,
                               data_generator,
                               (train_mean, train_std, train_istd),
                               save_root=args.write_dir)

    model: CausalFlowModel = instantiate_model(
        args, *data_generator.trajectory_generator._dyn.dims())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.sched_patience,
        cooldown=args.sched_cooldown,
        factor=1. / args.sched_factor)

    mse_loss = torch.nn.MSELoss().to(device)

    early_stop = EarlyStopping(es_patience=args.es_patience,
                               es_delta=args.es_delta)

    train_time = training_loop(experiment,
                               model,
                               mse_loss,
                               optimizer,
                               sched,
                               early_stop,
                               *data.get_loaders(args.batch_size),
                               device,
                               max_epochs=args.n_epochs)

    print(f"Training took {train_time:.2f} seconds.")
