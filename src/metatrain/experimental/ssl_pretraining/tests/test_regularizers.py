import copy
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TermsHypers
from metatrain.experimental.ssl_pretraining.objective import (
    build_step_data,
    grad_jepa_loss,
)
from metatrain.experimental.ssl_pretraining.regularizers import SigregTerm, sigreg
from metatrain.experimental.ssl_pretraining.terms import TERMS, build_terms
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError

from . import CARBON_PATH, QM9_PATH
from .test_objective import _setup


NAME = "experimental.ssl_pretraining"


def test_sigreg_small_for_gaussian_large_for_collapse():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    gauss = sigreg(torch.randn(2048, 32), 128, generator=g)
    collapsed = sigreg(
        torch.zeros(2048, 32) + 0.01 * torch.randn(2048, 32), 128, generator=g
    )
    assert collapsed > 20 * gauss


def test_sigreg_matches_a_direct_epps_pulley_evaluation():
    """The statistic of one slice is ``n`` times the weighted squared distance of the
    empirical characteristic function to the Gaussian one (trapezoidal rule)."""
    torch.manual_seed(0)
    x = torch.randn(64, 3, dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(3, 1, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm()
    t = torch.linspace(0, 3.0, 17, dtype=torch.float64)
    projected = (x @ direction)[:, 0, None] * t
    ecf = torch.complex(projected.cos().mean(0), projected.sin().mean(0))
    integrand = (ecf - torch.exp(-0.5 * t**2)).abs() ** 2 * torch.exp(-0.5 * t**2)
    expected = 64 * torch.trapezoid(integrand, t) * 2
    out = sigreg(x, 1, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(out, expected)


def test_sigreg_has_no_cost_and_stays_in_the_graph_below_two_rows():
    x = torch.randn(1, 8, requires_grad=True)
    out = sigreg(x, 16)
    assert float(out.detach()) == 0.0 and out.requires_grad


def test_sigreg_is_registered_with_its_hypers():
    assert TERMS["sigreg"] is SigregTerm
    assert "sigreg" in TermsHypers.__annotations__
    options = copy.deepcopy(get_default_hypers(NAME))
    assert options["training"]["objective"]["terms"]["sigreg"] is None
    options["training"]["objective"]["terms"]["sigreg"] = {
        "weight": 0.05,
        "n_slices": 64,
        "t_max": 3.0,
        "n_points": 17,
    }
    check_architecture_options(name=NAME, options=options)
    options["training"]["objective"]["terms"]["sigreg"]["n_points"] = 16
    with pytest.raises(MetatrainValidationError, match="n_points"):
        check_architecture_options(name=NAME, options=options)


def test_sigreg_term_reads_the_rows_of_vicreg_in_a_grad_jepa_step():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6"), torch.float64
    )
    step = build_step_data(
        student,
        teacher,
        None,
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(0),
        needs_gradients=False,
    )
    assert bool(step.views.masked.any())
    (term,) = build_terms({"sigreg": {"weight": 0.5, "n_slices": 32}})
    assert not term.needs_gradients and not term.uses_prediction
    # the slices are drawn from the step generator
    expected = sigreg(
        step.student_features[~step.views.masked],
        32,
        generator=torch.Generator().manual_seed(123),
    )
    step.extras["generator"] = torch.Generator().manual_seed(123)
    loss, diag = term(step)
    torch.testing.assert_close(loss, 0.5 * expected)
    torch.testing.assert_close(diag["sigreg/loss"], expected.detach())


def test_sigreg_contract_in_grad_jepa():
    """Alone in the Grad-JEPA objective: finite on degenerate batches, no
    warnings, trains the student only."""
    frames = [
        ase.Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.io.read(CARBON_PATH, 0),
    ]
    active = build_terms({"sigreg": {"weight": 0.05}})
    for batch in [frames[:1], frames[:2], frames]:
        student, teacher, _, systems, options, hypers = _setup(batch, torch.float64)
        for seed in range(2):
            student.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    None,
                    active,
                    systems,
                    options,
                    hypers,
                    torch.Generator().manual_seed(seed),
                    with_rank=True,
                )
                loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            assert any(p.grad is not None for p in student.parameters())
            assert all(p.grad is None for p in teacher.parameters())
