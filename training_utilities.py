import torch
import plotting_utilities as pu


def corr_loss(y_hat, y, eps=1e-8):
    y_hat = y_hat - y_hat.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    y_hat = y_hat / (y_hat.std(dim=0, keepdim=True).clamp_min(eps))
    y = y / (y.std(dim=0, keepdim=True).clamp_min(eps))
    corr = (y_hat * y).mean(dim=0).mean()
    return 1.0 - corr


def train_model(model, inputs, target, model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4, use_bfgs=False,
                clip_grad_norm=None, clip_grad_value=None, readout_xy=None):
    # --- train the model ---
    if use_bfgs:
        opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    else:
        opt = torch.optim.Adam(model.parameters(), lr=alpha)

    loss_list = []
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

        # log once per outer step
        loss_list.append(loss.item())

        print(f'step {step + 1} / {num_steps} | loss = {loss.item()}')

        if (step % save_every == 0) | (step == num_steps - 1):
            with torch.no_grad():
                if model_save_folder:
                    model.save(model_save_folder / 'model.pt')
                pred, readout = model.predict_and_readout(inputs)

                readout_names = list(model.readout_neurons.keys())
                pu.plot_readout_target(readout, target, readout_names=readout_names, save_folder=model_save_folder)

                if readout_xy is not None:
                    for i in range(readout.shape[2]):
                        pu.plot_response_over_space(readout[:4:, :, i].detach().cpu().numpy(), readout_xy[:, 0], readout_xy[:, 1], save_folder=model_save_folder / ('readout' + str(i)))

                pu.plot_loss(loss_list, save_folder=model_save_folder)


