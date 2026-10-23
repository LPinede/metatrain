import pytest
import torch

from metatrain.experimental.ssl_pretraining.predictor import GradJEPAPredictor
from metatrain.experimental.ssl_pretraining.views import random_rotations

from .conftest import D, N


def _predictor(
    species_query: bool = False, num_layers: int = 1, conditioners=()
) -> GradJEPAPredictor:
    torch.manual_seed(1)
    return GradJEPAPredictor(
        feature_dim=D,
        hidden_dim=16,
        probe_dim=4,
        num_rbf=8,
        cutoff=4.0,
        atomic_types=[1, 6],
        species_query=species_query,
        num_layers=num_layers,
        conditioners=conditioners,
    ).double()


def test_shapes(inputs):
    out = _predictor()(**inputs)
    assert out["latent"].shape == (N, D)
    assert out["gradient"].shape == (2, N, 3)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("improper", [False, True])
def test_rotation_covariance(inputs, improper, num_layers):
    predictor = _predictor(num_layers=num_layers)
    generator = torch.Generator().manual_seed(0)
    rotation = random_rotations(1, "SO3", torch.float64, generator)[0]
    if improper:
        # inversion composed with a rotation: det = -1
        rotation = -rotation
        assert torch.det(rotation) < 0
    else:
        assert torch.det(rotation) > 0
    rotated = dict(inputs)
    for key in ["gradients", "displacements", "vectors"]:
        rotated[key] = inputs[key] @ rotation.T
    out, out_rotated = predictor(**inputs), predictor(**rotated)
    torch.testing.assert_close(out["gradient"] @ rotation.T, out_rotated["gradient"])
    torch.testing.assert_close(out["latent"], out_rotated["latent"])


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_linear_in_probes(inputs, num_layers):
    """Student gradients are linear in the probes (g = J^T v), and so is the
    gradient prediction."""
    predictor = _predictor(num_layers=num_layers)
    scaled = dict(inputs)
    scaled["probes"] = 3.0 * inputs["probes"]
    scaled["gradients"] = 3.0 * inputs["gradients"]
    torch.testing.assert_close(
        3.0 * predictor(**inputs)["gradient"], predictor(**scaled)["gradient"]
    )


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_masked_features_are_ignored(inputs, num_layers):
    predictor = _predictor(num_layers=num_layers)
    changed = dict(inputs)
    changed["features"] = inputs["features"].clone()
    changed["features"][inputs["masked"]] += 10.0
    out, out_changed = predictor(**inputs), predictor(**changed)
    torch.testing.assert_close(out["latent"], out_changed["latent"])
    torch.testing.assert_close(out["gradient"], out_changed["gradient"])


def test_species_query(inputs):
    hidden, visible = _predictor(False), _predictor(True)
    visible.load_state_dict(hidden.state_dict())
    torch.nn.init.normal_(visible.species_embedding.weight)
    masked = inputs["masked"]
    out_hidden, out_visible = hidden(**inputs), visible(**inputs)
    assert not torch.allclose(
        out_hidden["latent"][masked], out_visible["latent"][masked]
    )
    # context-atom queries do not use the species embedding
    context_only = ~masked
    context_only[torch.unique(inputs["pairs"][masked[inputs["pairs"][:, 1]], 0])] = (
        False
    )
    torch.testing.assert_close(
        out_hidden["latent"][context_only], out_visible["latent"][context_only]
    )


# Outputs of the predictor before ``num_layers`` and the conditioner channels were
# added, for the ``inputs`` fixture and ``_predictor(species_query)``.
REFERENCE = {
    False: (
        -4.334189949934441,
        2.680093200478667,
        2.8585028329437,
        0.620090029290387,
        [-0.005571084694920488, 0.20456036459855345, -0.15123813120270055],
        [0.09423244392242641, 0.20072459925365996, -0.1709059030739738],
    ),
    True: (
        -4.088693803450036,
        2.641309596231314,
        2.8348897244829327,
        0.6118283337978343,
        [-0.005571084694920488, 0.20456036459855345, -0.15123813120270055],
        [0.09423244392242641, 0.20072459925365996, -0.1709059030739738],
    ),
}


@pytest.mark.parametrize("species_query", [False, True])
def test_default_depth_preserves_the_previous_outputs(inputs, species_query):
    out = _predictor(species_query, num_layers=1, conditioners=())(**inputs)
    latent_sum, gradient_sum, latent_sq, gradient_sq, latent_row, gradient_row = (
        REFERENCE[species_query]
    )
    kwargs = {"rtol": 1e-12, "atol": 1e-14}
    torch.testing.assert_close(out["latent"].sum().item(), latent_sum, **kwargs)
    torch.testing.assert_close(out["gradient"].sum().item(), gradient_sum, **kwargs)
    torch.testing.assert_close(out["latent"].pow(2).sum().item(), latent_sq, **kwargs)
    torch.testing.assert_close(
        out["gradient"].pow(2).sum().item(), gradient_sq, **kwargs
    )
    torch.testing.assert_close(out["latent"][0, :3].tolist(), latent_row, **kwargs)
    torch.testing.assert_close(out["gradient"][1, 5].tolist(), gradient_row, **kwargs)


def test_first_layer_parameters_do_not_depend_on_the_depth():
    one = _predictor(num_layers=1).state_dict()
    deep = _predictor(num_layers=3)
    assert len(deep.context_layers) == 2
    for key, value in one.items():
        torch.testing.assert_close(deep.state_dict()[key], value)


def test_context_layers_extend_the_receptive_field(inputs):
    """Atom 9 is beyond the cutoff of atom 5 (both context atoms): only the
    context update among context atoms lets it reach atom 5."""
    changed = dict(inputs)
    changed["features"] = inputs["features"].clone()
    changed["features"][9] += 1.0
    shallow, deep = _predictor(num_layers=1), _predictor(num_layers=2)
    # exactly unchanged (atom 9 is never read) versus a small but nonzero change
    shift = shallow(**inputs)["latent"][5] - shallow(**changed)["latent"][5]
    assert torch.equal(shift, torch.zeros_like(shift))
    shift = deep(**inputs)["latent"][5] - deep(**changed)["latent"][5]
    assert shift.abs().max() > 0


def test_invalid_depth():
    with pytest.raises(ValueError, match="num_layers"):
        _predictor(num_layers=0)
