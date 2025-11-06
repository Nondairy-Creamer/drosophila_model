import torch
import plotting_utilities as pu
import numpy as np
import neural_models as nm


def corr_loss(y_hat, y, neuron_scaling=None, eps=1e-6):
    # Compute a single correlation across all elements by normalizing globally
    y_hat = y_hat - y_hat.mean(dim=(1, 2), keepdim=True)
    y = y - y.mean(dim=(1, 2), keepdim=True)
    y_hat = y_hat / (y_hat.std(dim=(1, 2), keepdim=True).clamp_min(eps))
    y = y / (y.std(dim=(1, 2), keepdim=True).clamp_min(eps))
    corr = (y_hat * y).mean(dim=(1, 2), keepdim=True)

    if neuron_scaling is not None:
        assert torch.all(neuron_scaling >= 0) & ~torch.all(neuron_scaling == 0), 'contrast scaling cant be negative or all zeros'
        neuron_scaling = neuron_scaling / neuron_scaling.sum()
        corr = torch.sum(corr * neuron_scaling[None, None, None, :], dim=-1)
        
    ave_corr = corr.mean()

    return 1.0 - ave_corr

def cos_loss(y_hat, y, neuron_scaling=None, eps=1e-6):
    # Compute a single correlation across all elements by normalizing globally
    y_hat = y_hat / y_hat.norm(dim=(1, 2), keepdim=True).clamp_min(eps)
    y = y / y.norm(dim=(1, 2), keepdim=True).clamp_min(eps)
    cosine_sim = (y_hat * y).sum(dim=(1, 2), keepdim=True)

    if neuron_scaling is not None:
        assert torch.all(neuron_scaling >= 0) & ~torch.all(neuron_scaling == 0), 'contrast scaling cant be negative or all zeros'
        neuron_scaling = neuron_scaling / neuron_scaling.sum()
        cosine_sim = torch.sum(cosine_sim * neuron_scaling[None, None, None, :], dim=-1)
        
    ave_cosine_sim = cosine_sim.mean()

    return 1.0 - ave_cosine_sim

def combined_loss(y_hat, y, neuron_scaling=None, eps=1e-6):
    cos_loss_val = cos_loss(y_hat, y, neuron_scaling=neuron_scaling, eps=eps)
    corr_loss_val = corr_loss(y_hat, y, neuron_scaling=neuron_scaling, eps=eps)
    return 0.5 * cos_loss_val + 0.5 * corr_loss_val

def get_scaled_loss(loss_fn, neuron_scaling=None):
    def loss_out(y_hat, y):
        return loss_fn(y_hat, y, neuron_scaling=neuron_scaling)
    return loss_out

def train_model(model, inputs, target, model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None, readout_xy=None, reload_best_on_loss_jump=True, loss_fn=combined_loss,
                patience=100, alpha_factor=0.5, opt_state_dict=None, loss_history=None, stim_to_cell_mapping=None):
    # --- train the model ---
    opt = torch.optim.Adam(model.parameters(), lr=alpha)

    if opt_state_dict is not None:
        opt.load_state_dict(opt_state_dict)

    if loss_history is None:
        loss_list = []
        best_loss = np.inf
    else:
        loss_list = loss_history
        best_loss = loss_history[-1]

    time_since_best = 0
    grad_norm = 0

    for step in range(num_steps):
        opt.zero_grad()
        readout = model.predict_and_readout(inputs, stim_to_cell_mapping)[1]
        loss = loss_fn(readout, target)

        # Detect if you've found the best loss so far BEFORE updating weights
        loss_list.append(loss.item())
        if loss_list[-1] < best_loss:
            best_loss = loss_list[-1]
            time_since_best = 0

            # track and save the best checkpoint for easy rollback
            # Save BEFORE opt.step() so we save the weights that achieved this loss
            if model_save_folder:
                model.save(model_save_folder / 'model.pt')
                torch.save({
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss_list,
                    'step': step
                }, model_save_folder / 'model_opt.pt')
        else:
            time_since_best += 1
            
        # Update weights AFTER checking/saving best model
        loss.backward()

        # Optional gradient clipping (prefer norm if both provided)
        if clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        elif clip_grad_value is not None:
            grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
        opt.step()

        loss_diff = 0
        if len(loss_list) > 1:
            loss_diff = loss_list[-2] - loss_list[-1]
        print(f'step {step + 1} / {num_steps} | loss = {loss.item():4g} | loss diff = {loss_diff:.4e} | lr = {alpha:.2e} | grad magnitude = {grad_norm:.3g} | path = {model_save_folder}')

        if reload_best_on_loss_jump and (model_save_folder is not None):
            if time_since_best > patience:
                # Load the best model state into the existing model
                checkpoint = torch.load(model_save_folder / 'model.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
                
                # Load the best optimizer state and reduce learning rate
                opt_state = torch.load(model_save_folder / 'model_opt.pt', map_location='cpu', weights_only=False)
                opt.load_state_dict(opt_state['optimizer_state_dict'])
                
                alpha = alpha * alpha_factor
                for param_group in opt.param_groups:
                    param_group['lr'] = alpha
                    
                time_since_best = 0
                print(' -------------------------------- ')
                print(f'Reloaded best checkpoint. Learning rate reduced to {alpha:.2e}.')
                print(' -------------------------------- ')

        if (step % save_every == 0) | (step == num_steps - 1):
            with torch.no_grad():
                readout_names = list(model.readout_indexes.keys())
                pu.plot_readout_target(readout, target, readout_names=readout_names, save_folder=model_save_folder)

                if readout_xy is not None:
                    for i in range(readout.shape[0]):
                        pu.plot_response_over_space(readout[i, ::4, :, :].detach().cpu().numpy().mean(axis=1), readout_xy[:, 0], readout_xy[:, 1], save_folder=model_save_folder / ('readout' + str(i)))

                pu.plot_loss(loss_list, save_folder=model_save_folder)



