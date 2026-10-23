import ase
import pytest
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.grad_jepa.predictor import GradJEPAPredictor
from metatrain.experimental.grad_jepa.views import pair_vectors, random_rotations
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import to_systems


D = 12
N = 10


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    options = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)
    chain = ase.Atoms(
        "C5H5", positions=[[1.2 * i, 0.3 * (i % 2), 0.0] for i in range(N)]
    )
    (system,) = to_systems([chain], torch.float64)
    system = get_system_with_neighbor_lists(system, [options])
    samples = system.get_neighbor_list(options).samples.values
    vectors = pair_vectors(system.positions, system.cell, samples)
    close = vectors.norm(dim=-1) < 4.0
    masked = torch.zeros(N, dtype=torch.bool)
    masked[3:5] = True
    gradients = torch.randn(2, N, 3, dtype=torch.float64)
    gradients[:, masked] = 0.0
    return {
        "features": torch.randn(N, D, dtype=torch.float64),
        "gradients": gradients,
        "probes": torch.randn(2, D, dtype=torch.float64),
        "displacements": 0.05 * torch.randn(N, 3, dtype=torch.float64),
        "types": system.types,
        "masked": masked,
        "pairs": samples[close, :2],
        "vectors": vectors[close],
    }


def _predictor(species_query: bool = False) -> GradJEPAPredictor:
    torch.manual_seed(1)
    return GradJEPAPredictor(
        feature_dim=D,
        hidden_dim=16,
        probe_dim=4,
        num_rbf=8,
        cutoff=4.0,
        atomic_types=[1, 6],
        species_query=species_query,
    ).double()


def test_shapes(inputs):
    out = _predictor()(**inputs)
    assert out["latent"].shape == (N, D)
    assert out["gradient"].shape == (2, N, 3)


def test_rotation_covariance(inputs):
    predictor = _predictor()
    rotation = random_rotations(1, "O3", torch.float64, torch.Generator())[0]
    rotated = dict(inputs)
    for key in ["gradients", "displacements", "vectors"]:
        rotated[key] = inputs[key] @ rotation.T
    out, out_rotated = predictor(**inputs), predictor(**rotated)
    torch.testing.assert_close(out["gradient"] @ rotation.T, out_rotated["gradient"])
    torch.testing.assert_close(out["latent"], out_rotated["latent"])


def test_linear_in_probes(inputs):
    """Student gradients are linear in the probes (g = J^T v), and so is the
    gradient prediction."""
    predictor = _predictor()
    scaled = dict(inputs)
    scaled["probes"] = 3.0 * inputs["probes"]
    scaled["gradients"] = 3.0 * inputs["gradients"]
    torch.testing.assert_close(
        3.0 * predictor(**inputs)["gradient"], predictor(**scaled)["gradient"]
    )


def test_masked_features_are_ignored(inputs):
    predictor = _predictor()
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
