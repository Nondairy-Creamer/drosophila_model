import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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

                plt.figure()
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
                direction = ['progressive', 'regressive', 'up', 'down']
                readout_plot = readout.detach().cpu().numpy().T.reshape((4, readout.shape[0] // 4)).T
                target_plot = target.detach().cpu().numpy().T.reshape((4, target.shape[0] // 4)).T
                for i in range(readout_plot.shape[1]):
                    plt.plot(target_plot[:, i], color=colors[i], label=direction[i] + ' target')
                    plt.plot(readout_plot[:, i], color=colors[i], ls='--', label=direction[i] + ' prediction')
                plt.legend()
                plt.savefig('figures/pred_vs_target.png')
                plt.close()

                plt.figure()
                plt.plot(loss_list, label='loss')
                plt.legend()
                plt.savefig('figures/loss.png')
                plt.close()
