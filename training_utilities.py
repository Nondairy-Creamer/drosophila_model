import torch
import plotting_utilities as pu


def train_model(model, inputs, target, model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4, use_bfgs=False,
                clip_grad_norm=None, clip_grad_value=None):
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
            loss = torch.nn.functional.mse_loss(readout, target)
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
                pu.plot_loss(loss, save_folder=model_save_folder)


