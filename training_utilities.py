import torch
import plotting_utilities as pu


def corr_loss(y_hat, y, eps=1e-8):
    # Compute a single correlation across all elements by normalizing globally
    y_hat = y_hat - y_hat.mean()
    y = y - y.mean()
    y_hat = y_hat / (y_hat.std().clamp_min(eps))
    y = y / (y.std().clamp_min(eps))
    corr = (y_hat * y).mean()
    return 1.0 - corr


def train_model(model, inputs, target, model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4, use_bfgs=False,
                clip_grad_norm=None, clip_grad_value=None, readout_xy=None, lr_plateau_start_loss=0.99,
                reload_best_on_lr_drop=True):
    # --- train the model ---
    if use_bfgs:
        opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    else:
        opt = torch.optim.Adam(model.parameters(), lr=alpha)

    # Reduce LR on plateau in training loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=30, threshold=1e-8,
        threshold_mode='rel', cooldown=10, min_lr=1e-6,
    )

    loss_list = []
    prev_lr = opt.param_groups[0]['lr']
    for step in range(num_steps):
        def closure():
            opt.zero_grad()
            readout = model.predict_and_readout(inputs)[1]
            # loss = torch.nn.functional.mse_loss(readout, target)
            loss = corr_loss(readout, target)
            loss.backward()
            # Optional gradient clipping (prefer norm if both provided)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            elif clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
            return loss

        if use_bfgs:
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()

        # step the LR scheduler with the latest loss value, optionally gating start
        loss_value = loss.item()
        if (lr_plateau_start_loss is None) or (loss_value <= lr_plateau_start_loss):
            scheduler.step(loss_value)

        # Detect LR drop (plateau) and optionally reload the best checkpoint
        new_lr = opt.param_groups[0]['lr']
        if reload_best_on_lr_drop and (new_lr < prev_lr) and (model_save_folder is not None):
            best_path = (model_save_folder / 'best_model.pt')
            if best_path.exists():
                # load only the state_dict to the current model
                data = torch.load(best_path, map_location='cpu', weights_only=False)
                model.load_state_dict(data['state_dict'])
                # reinitialize optimizer to clear stale state, keep the new LR
                if use_bfgs:
                    opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
                else:
                    opt = torch.optim.Adam(model.parameters(), lr=new_lr)
                print(' -------------------------------- ')
                print(f'Learning rate reduced to {new_lr:.2e}. Reloaded best checkpoint from {best_path.name}.')
                print(' -------------------------------- ')
        prev_lr = new_lr

        # log once per outer step
        loss_list.append(loss.item())

        if len(loss_list) > 1:
            rel_change = (loss_list[-2] - loss_list[-1]) / loss_list[-1] if loss_list else 0
        else:
            rel_change = 0

        print(f'step {step + 1} / {num_steps} | loss = {loss.item()} | rel_change = {rel_change} | lr = {new_lr}')

        if (step % save_every == 0) | (step == num_steps - 1):
            with torch.no_grad():
                if model_save_folder:
                    # always save a rolling latest checkpoint
                    model.save(model_save_folder / 'model.pt')
                    # save a periodic checkpoint with step and loss for recovery
                    ckpt_name = f"model_step_{step+1:07d}_loss_{loss_value:.6f}.pt"
                    model.save(model_save_folder / ckpt_name)
                pred, readout = model.predict_and_readout(inputs)

                readout_names = list(model.readout_neurons.keys())
                pu.plot_readout_target(readout, target, readout_names=readout_names, save_folder=model_save_folder)

                if readout_xy is not None:
                    for i in range(readout.shape[0]):
                        pu.plot_response_over_space(readout[i, :4:, :].detach().cpu().numpy(), readout_xy[:, 0], readout_xy[:, 1], save_folder=model_save_folder / ('readout' + str(i)))

                pu.plot_loss(loss_list, save_folder=model_save_folder)

        # track and save the best checkpoint for easy rollback
        if model_save_folder:
            if (len(loss_list) == 1) or (loss_value < min(loss_list[:-1])):
                model.save(model_save_folder / 'best_model.pt')


