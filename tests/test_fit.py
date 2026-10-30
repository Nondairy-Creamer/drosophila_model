"""Training-path tests: dataset generation, and one-step ``fit``
runs on the small test model with one- and two-eye data."""
import math

import numpy as np
import pytest
import torch

from drosophila_model.config import load_config
from drosophila_model.data_handling import stimulus_generators as sg
from drosophila_model.training import training_utilities as tu
from tests.model_factory import build_model_from_test_config

# Right-eye camera offset of a two-camera rig (metres); the left eye mirrors it.
EYE_OFFSET = (0.05, -0.05, 0.013)


def _dataset(model, num_eyes, num_stimuli=2, num_time=4):
    """Random inputs and targets laid out like a training dataset."""
    gen = torch.Generator().manual_seed(0)
    num_cols = num_eyes * model.regions['optic_lobe'].num_columns

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=model.dtype).to(model.device)

    inputs = randn(num_stimuli, num_time, num_cols)
    target = {'optic_flow': 0.1 * randn(num_stimuli, num_time, num_cols, 2)}
    if num_eyes == 2:
        target['rotation'] = 0.1 * randn(num_stimuli, num_time, 3)
        target['translation'] = randn(num_stimuli, num_time, 3)
    train_mask = torch.ones(inputs.shape, dtype=torch.bool, device=model.device)
    return inputs, target, train_mask


def _train_one_step(model, num_eyes, loss_components, **kwargs):
    inputs, target, train_mask = _dataset(model, num_eyes)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    _, loss_history, _, info = tu.fit(
        model, inputs, target,
        output_neurons=None, input_neurons=None,
        loss_components=loss_components,
        rng=np.random.default_rng(0),
        num_epochs=1, save_every=1, alpha=1e-3,
        train_mask=train_mask, num_eyes=num_eyes,
        **kwargs,
    )
    changed = [k for k, v in model.named_parameters() if not torch.equal(v.detach(), initial[k])]
    return loss_history, info, changed


@pytest.mark.parametrize("dynamics_type", ["CurrentDynamics", "ConductanceDynamics"])
def test_train_one_eye_flow_loss(tmp_path, dynamics_type):
    model, _ = build_model_from_test_config(
        readouts=['FlowConv2Readout', 'L1Reg'], dynamics_type=dynamics_type)
    run_dir = tmp_path / "run"

    loss_history, info, changed = _train_one_step(
        model, 1, ['flow_loss'], model_save_folder=run_dir)

    # The regularizer readout is logged next to the configured loss.
    assert set(loss_history) == {'flow_loss', 'L1Reg'}
    assert all(len(v) == 1 and math.isfinite(v[0]) for v in loss_history.values())
    assert math.isfinite(info['best_loss'])
    assert changed, "one optimizer step changed no parameters"
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "model_opt.pt").exists()
    assert (run_dir / "figures" / "training" / "loss.png").exists()


@pytest.mark.parametrize("history_best, saves", [(1e-12, False), (1e12, True)])
def test_resumed_run_saves_only_on_improvement(tmp_path, history_best, saves):
    # Resuming with a loss history: model.pt already holds that history's best
    # epoch, so it is only overwritten once a new epoch beats it.
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    run_dir = tmp_path / "run"

    loss_history, info, _ = _train_one_step(
        model, 1, ['flow_loss'], model_save_folder=run_dir,
        loss_history={'flow_loss': [history_best]})

    assert len(loss_history['flow_loss']) == 2
    assert (run_dir / "model.pt").exists() == saves
    if not saves:
        assert info['best_loss'] == pytest.approx(history_best)


def test_train_two_eye_flow_loss():
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])

    loss_history, _, changed = _train_one_step(
        model, 2, ['flow_loss'], eye_offset=EYE_OFFSET)

    assert list(loss_history) == ['flow_loss']
    assert math.isfinite(loss_history['flow_loss'][0])
    assert changed


def test_train_two_eye_odometry_loss():
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    loss_components = ['flow_loss', 'odometry_loss']

    # The training forward pass, reproduced by the shared evaluation helpers.
    inputs, target, train_mask = _dataset(model, 2)
    with torch.no_grad():
        outputs, _ = tu.model_outputs(
            model, inputs, train_mask, num_eyes=2, eye_offset=EYE_OFFSET)
        components = tu.compute_loss_components(
            model, outputs, target, train_mask, loss_components)
    assert outputs['pred_rotation'].shape == target['rotation'].shape
    assert outputs['pred_translation'].shape == target['translation'].shape
    assert outputs['FlowConv2Readout'].shape == target['optic_flow'].shape
    assert set(components) == set(loss_components)
    assert all(torch.isfinite(v) for v in components.values())

    loss_history, _, changed = _train_one_step(
        model, 2, loss_components, eye_offset=EYE_OFFSET)

    assert set(loss_history) == set(loss_components)
    assert all(math.isfinite(v[0]) for v in loss_history.values())
    assert changed


@pytest.mark.parametrize("loss_components, num_eyes, match", [
    (['not_a_loss'], 1, "Unknown loss component"),
    (['odometry_loss'], 1, "requires a two-eye dataset"),
    (['flow_loss'], 3, "num_eyes must be 1 or 2"),
])
def test_fit_rejects_bad_settings(loss_components, num_eyes, match):
    # Settings are validated before the model or data are touched.
    with pytest.raises(ValueError, match=match):
        tu.fit(None, None, None, None, None, loss_components, num_eyes=num_eyes)


def _fake_stimulus(stim_type, config, column_phi_theta, num_eyes=None):
    """A tiny ``StimulusData``: each example's stimulus holds its ``group_id``.

    ``random_pano`` has 4 groups of 2 examples and the other types 4 groups of
    1, so types concatenate; ``random_sine`` is one-eye.
    """
    group_id = torch.arange(4).repeat_interleave(2 if stim_type == 'random_pano' else 1)
    num_eyes = 1 if stim_type == 'random_sine' else 2
    num_examples, num_time, num_cols = len(group_id), 3, 5 * num_eyes
    stimulus = group_id[:, None, None].expand(num_examples, num_time, num_cols).to(torch.float64)
    two_eye = num_eyes == 2
    return sg.StimulusData(
        stimulus=stimulus,
        optic_flow=torch.rand(num_examples, num_time, num_cols, 2),
        train_mask=torch.ones(num_examples, num_time, num_cols, dtype=torch.bool),
        rotation=torch.rand(num_examples, num_time, 3) if two_eye else None,
        translation=torch.rand(num_examples, num_time, 3) if two_eye else None,
        num_eyes=num_eyes,
        eye_offset=EYE_OFFSET if two_eye else None,
        group_id=group_id,
    )


def test_generate_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(tu.sg, 'get_stimulus', _fake_stimulus)
    monkeypatch.setattr(tu, 'load_column_phi_theta', lambda folder: None)
    config = {
        'model': {'device': 'cpu', 'dtype': 'float32'},
        'stimulus': {'stimulus_type': ['random_pano', 'static_sine']},
        'data': {
            'data_folder': str(tmp_path),
            'shared': {'seed': 1, 'train_fraction': 0.75},
            'full_field_step': {'train_fraction': 1.0},
        },
    }

    data = tu.generate_dataset(config)
    assert data['stimulus'].shape == (9, 3, 10)
    assert data['stimulus'].dtype == torch.float32
    assert data['optic_flow'].shape == (9, 3, 10, 2)
    assert data['rotation'].shape == data['translation'].shape == (9, 3, 3)
    assert data['num_eyes'] == 2
    assert data['eye_offset'] == EYE_OFFSET

    # The split is by whole group: 3 of 4 groups train, and none straddles.
    config['stimulus']['stimulus_type'] = 'random_pano'
    train = tu.generate_dataset(config)['stimulus'][:, 0, 0]
    test = tu.generate_dataset(config, split='test')['stimulus'][:, 0, 0]
    assert len(train) == 6 and len(test) == 2
    assert not set(train.tolist()) & set(test.tolist())
    assert torch.equal(train, tu.generate_dataset(config)['stimulus'][:, 0, 0])

    config['stimulus']['stimulus_type'] = 'random_sine'
    data = tu.generate_dataset(config)
    assert data['num_eyes'] == 1 and data['eye_offset'] is None
    assert data['rotation'] is None and data['translation'] is None

    config['stimulus']['stimulus_type'] = ['random_pano', 'random_sine']
    with pytest.raises(ValueError, match="Mixed eye counts"):
        tu.generate_dataset(config)
    with pytest.raises(ValueError, match="split"):
        tu.generate_dataset(config, split='validation')

    # Analysis-only and unknown types are refused.
    config['stimulus']['stimulus_type'] = 'edge'
    with pytest.raises(ValueError, match="cannot be used for training"):
        tu.generate_dataset(config)
    config['stimulus']['stimulus_type'] = 'not_a_stimulus'
    with pytest.raises(ValueError, match="Unknown stimulus_type"):
        tu.generate_dataset(config)

    # A train-only type is skipped in the test split, unless it is the only one.
    config['stimulus']['stimulus_type'] = ['random_pano', 'full_field_step']
    assert tu.generate_dataset(config, split='test')['stimulus'].shape[0] == 2
    assert tu.generate_dataset(config)['stimulus'].shape[0] == 10
    config['stimulus']['stimulus_type'] = 'full_field_step'
    with pytest.raises(ValueError, match="training split"):
        tu.generate_dataset(config, split='test')


def test_generate_dataset_real_generator(tmp_path, monkeypatch):
    """The real generators are deterministic, and the split keeps their groups apart."""
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    config = load_config(str(empty), str(empty))
    config['model']['device'] = 'cpu'
    config['data']['data_folder'] = str(tmp_path)
    config['stimulus']['stimulus_type'] = 'counterphase_sine'
    columns = np.column_stack([np.linspace(-1.0, 1.0, 5), np.full(5, np.pi / 2)])
    monkeypatch.setattr(tu, 'load_column_phi_theta', lambda folder: columns)

    train = tu.generate_dataset(config)
    test = tu.generate_dataset(config, split='test')
    assert train['stimulus'].shape[2] == 10 and train['num_eyes'] == 2
    assert train['stimulus'].shape[0] == 9 and test['stimulus'].shape[0] == 3
    assert torch.equal(train['stimulus'], tu.generate_dataset(config)['stimulus'])


@pytest.mark.parametrize("dynamics_type", [
    "CurrentDynamics", "CurrentDivisiveDynamics", "ConductanceDynamics",
    "CellAnnDynamics", "ArbitraryDynamics",
])
def test_every_dynamics_lifecycle(tmp_path, dynamics_type):
    """Build, train a step, save/load, silence (and restore), and trim."""
    from drosophila_model.model_definition import neural_models as nm

    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'],
                                            dynamics_type=dynamics_type)
    _, _, changed = _train_one_step(model, 2, ['flow_loss'], eye_offset=EYE_OFFSET)
    assert changed

    path = tmp_path / "model.pt"
    model.save(path)
    loaded = nm.BaseRnn.load(path, device='cpu')
    for (name, p), (_, q) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(p.cpu(), q.cpu()), name

    before = {k: v.clone() for k, v in model.state_dict().items()}
    with model.silenced(['Mi1'], silence_inputs=True):
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    trimmed = model.remove_cell_types(['Mi1'])
    assert 'Mi1' not in trimmed.unique_cell_types
    inputs, _, _ = _dataset(trimmed, 1)
    assert torch.isfinite(trimmed.predict(inputs)['outputs']['optic_lobe']['FlowConv2Readout']).any()
