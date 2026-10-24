import copy
from dataclasses import fields

import ase.io
import numpy as np
import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.ssl_pretraining.high_order import views as ho_views
from metatrain.experimental.ssl_pretraining.high_order.deformations import (
    shear_matrix,
    smooth_field_with_gradient,
)
from metatrain.experimental.ssl_pretraining.high_order.views import (
    HighOrderExtras,
    build_high_order_views,
)
from metatrain.experimental.ssl_pretraining.objective import (
    RESERVED_EXTRAS,
    VIEW_BUILDERS,
)
from metatrain.experimental.ssl_pretraining.tests import QM9_PATH
from metatrain.experimental.ssl_pretraining.views import pair_vectors

from . import make_batch


def _views(batch, seed=0, position_gradients=True, **over):
    systems, options, hypers = batch
    hypers = copy.deepcopy(hypers)
    hypers["high_order"].update(over)
    generator = torch.Generator().manual_seed(seed)
    views = build_high_order_views(
        systems, options, hypers, generator, position_gradients
    )
    return systems, options, views


def _min_image(vectors, cell, pbc):
    """Minimum-image version of ``vectors`` (unchanged without periodicity)."""
    if not bool(pbc.any()):
        return vectors
    frac = vectors @ torch.linalg.inv(cell)
    return (frac - torch.round(frac)) @ cell


def test_registered_on_import():
    assert VIEW_BUILDERS["high_order"] is build_high_order_views


def test_student_frame_is_rotated_teacher_frame(small_batch):
    systems, _, v = _views(small_batch, p_def=0.0)  # mask-only: no deformation
    ho = v.extras["high_order"]
    offset = 0
    for s, st in zip(systems, v.student, strict=True):
        rot = ho.rotations[offset]  # every atom carries its structure's rotation
        n = len(s)
        assert torch.allclose(ho.rotations[offset : offset + n], rot.expand(n, 3, 3))
        assert torch.allclose(st.positions, s.positions @ rot.T, atol=1e-6)
        offset += n
    assert torch.equal(ho.rotations, v.rotations)


def test_deformed_branch_bounds_and_targets(small_batch):
    systems, _, v = _views(small_batch, p_def=1.0)
    ho = v.extras["high_order"]
    skin = small_batch[2]["noise"]["skin"]
    assert ho.u_student.norm(dim=-1).max() <= skin / 2 + 1e-6
    assert ho.u_student.abs().max() > 0
    assert ho.branch_deformed.all() and not ho.branch_masked.any()
    assert not v.masked.any()
    assert torch.allclose(
        ho.observed_vectors - ho.clean_vectors,
        ho.u_student[ho.pair_index[:, 1]] - ho.u_student[ho.pair_index[:, 0]],
        atol=1e-6,
    )
    offset = 0
    for s, st in zip(systems, v.student, strict=True):
        n = len(s)
        u = ho.u_student[offset : offset + n]
        rot = ho.rotations[offset]
        assert torch.allclose(st.positions, (s.positions + u) @ rot.T, atol=1e-6)
        offset += n


@pytest.mark.parametrize("p_def", [0.0, 1.0])
def test_student_pairs_follow_the_displacement(large_batch, p_def):
    """The student pair vectors (with cell shifts for the periodic crystal), mapped
    back to the teacher frame, are the displaced clean pair vectors."""
    systems, options, v = _views(large_batch, p_def=p_def)
    ho = v.extras["high_order"]
    offset, pair_offset = 0, 0
    for s, st in zip(systems, v.student, strict=True):
        n = len(s)
        atoms = slice(offset, offset + n)
        samples = s.get_neighbor_list(options).samples.values
        displaced = pair_vectors(s.positions + ho.u_student[atoms], s.cell, samples)
        masked = v.masked[atoms]
        keep = ~(masked[samples[:, 0]] | masked[samples[:, 1]])
        student_vectors = st.get_neighbor_list(options).values.reshape(-1, 3)
        rot = ho.rotations[offset]
        torch.testing.assert_close(student_vectors @ rot, displaced[keep])
        close = pair_vectors(s.positions, s.cell, samples).norm(dim=-1) < 4.0
        n_close = int(close.sum())
        rows = slice(pair_offset, pair_offset + n_close)
        torch.testing.assert_close(ho.observed_vectors[rows], displaced[close])
        offset += n
        pair_offset += n_close
    assert pair_offset == ho.pair_index.shape[0]


def test_masked_branch_has_no_student_deformation(large_batch):
    _, _, v = _views(large_batch, p_def=0.0)
    ho = v.extras["high_order"]
    assert ho.branch_masked.all() and not ho.branch_deformed.any()
    assert v.masked.any()
    assert torch.equal(ho.u_student, torch.zeros_like(ho.u_student))
    assert torch.equal(ho.u_smooth, torch.zeros_like(ho.u_smooth))
    assert torch.equal(ho.grad_smooth, torch.zeros_like(ho.grad_smooth))
    assert torch.equal(ho.observed_vectors, ho.clean_vectors)


def test_teacher_pair_is_antithetic(small_batch):
    systems, _, v = _views(small_batch, p_def=1.0)
    plus, minus = v.teacher, v.extras["teacher_views"]["minus"]  # teacher = plus
    assert set(v.extras["teacher_views"]) == {"minus"}
    skin = small_batch[2]["noise"]["skin"]
    moved = False
    for s, p, m in zip(systems, plus, minus, strict=True):
        assert torch.allclose(p.positions + m.positions, 2 * s.positions, atol=1e-6)
        u_t = (p.positions - s.positions).norm(dim=-1)
        assert u_t.max() <= skin / 2 + 1e-6
        moved = moved or bool(u_t.max() > 0)
    assert moved


def test_teacher_mode_is_the_sum_of_centre_modes(large_batch):
    """The teacher displacement is the windowed l=1 modes of the centres, with
    minimum-image relative positions in the periodic crystal."""
    systems, _, v = _views(large_batch, p_def=1.0, mode_amplitude=0.01)
    ho = v.extras["high_order"]
    window = large_batch[2]["high_order"]["mode_window"]
    offset = 0
    for s, p in zip(systems, v.teacher, strict=True):
        n = len(s)
        mine = (ho.centers >= offset) & (ho.centers < offset + n)
        expected = torch.zeros_like(s.positions)
        centers, amplitudes = ho.centers[mine] - offset, ho.center_amplitudes[mine]
        for c, a in zip(centers, amplitudes, strict=True):
            rel = _min_image(s.positions - s.positions[c], s.cell, s.pbc)
            w = torch.exp(-((rel.norm(dim=-1) / window) ** 2))[:, None]
            linear = (
                a[0] / 3**0.5 * rel
                + torch.cross(a[1:4].expand_as(rel), rel, dim=-1)
                + rel @ shear_matrix(a[None, 4:9])[0].T
            )
            expected = expected + w * linear
        torch.testing.assert_close(p.positions - s.positions, expected)
        offset += n


def test_centers_are_far_from_mask(large_batch):
    systems, options, v = _views(large_batch, p_def=0.0)
    ho = v.extras["high_order"]
    hypers = large_batch[2]["high_order"]
    assert ho.centers.numel() > 0
    # no centre has a masked atom within the cutoff
    near = ho.pair_index[ho.hidden_source, 0]
    assert not torch.isin(ho.centers, near).any()
    assert not v.masked[ho.centers].any()
    assert ho.context[ho.centers].all()
    # at least 3 neighbours within the cutoff (identifiable l=1 amplitudes)
    counts = torch.bincount(ho.pair_index[:, 0], minlength=len(v.masked))
    assert (counts[ho.centers] >= 3).all()
    offset = 0
    for s in systems:
        n = len(s)
        mine = ho.centers[(ho.centers >= offset) & (ho.centers < offset + n)] - offset
        assert len(mine) <= hypers["max_centers"]
        for a in range(len(mine)):
            for b in range(a):
                d = s.positions[mine[a]] - s.positions[mine[b]]
                assert _min_image(d, s.cell, s.pbc).norm() >= 2 * options.cutoff
        offset += n


def test_one_sided_teacher(small_batch):
    systems, _, v = _views(small_batch, p_def=1.0, teacher_pair=False)
    assert set(v.extras["teacher_views"]) == {"plus"}
    moved = False
    for s, t, p in zip(
        systems, v.teacher, v.extras["teacher_views"]["plus"], strict=True
    ):
        assert torch.equal(t.positions, s.positions)  # the clean x0
        moved = moved or bool((p.positions != s.positions).any())
    assert moved


def test_reproducible(large_batch):
    _, _, a = _views(large_batch, second_rotation=True)
    _, _, b = _views(large_batch, second_rotation=True)
    for f in fields(HighOrderExtras):
        assert torch.equal(
            getattr(a.extras["high_order"], f.name),
            getattr(b.extras["high_order"], f.name),
        ), f.name
    for sa, sb in zip(a.student, b.student, strict=True):
        assert torch.equal(sa.positions, sb.positions)
    _, _, c = _views(large_batch, seed=1)
    assert not torch.equal(
        a.extras["high_order"].rotations, c.extras["high_order"].rotations
    )


def test_tiny_structures_have_no_centers_and_no_nan(tiny_batch):
    for p_def in (0.0, 1.0):
        systems, _, v = _views(tiny_batch, p_def=p_def, second_rotation=True)
        ho = v.extras["high_order"]
        assert ho.centers.numel() == 0
        assert ho.center_amplitudes.shape == (0, 9)
        for f in fields(HighOrderExtras):
            value = getattr(ho, f.name)
            if value.is_floating_point():
                assert torch.isfinite(value).all(), f.name
        for s, t in zip(systems, v.teacher, strict=True):
            assert torch.equal(t.positions, s.positions)  # no centre: u_T = 0


@pytest.mark.parametrize("position_gradients", [True, False])
def test_position_gradients_honoured(small_batch, position_gradients):
    for teacher_pair in (True, False):
        _, _, v = _views(
            small_batch,
            position_gradients=position_gradients,
            teacher_pair=teacher_pair,
        )
        for system in v.student + v.teacher:
            assert system.positions.requires_grad == position_gradients
        for extra in v.extras["teacher_views"].values():
            for system in extra:
                assert not system.positions.requires_grad


def test_extras_keys_are_not_reserved(small_batch):
    _, _, v = _views(small_batch)
    assert set(v.extras) == {"high_order", "teacher_views"}
    assert not set(v.extras) & set(RESERVED_EXTRAS)
    assert isinstance(v.extras["high_order"], HighOrderExtras)


def test_views_fields(large_batch):
    systems, _, v = _views(large_batch, p_def=0.0)
    ho = v.extras["high_order"]
    n_atoms = sum(len(s) for s in systems)
    assert v.masked.shape == (n_atoms,) and v.masked.dtype == torch.bool
    # latent targets: masked atoms with a visible neighbour within the cutoff, and
    # every visible atom
    reachable = torch.zeros_like(v.masked)
    pairs = ho.pair_index
    reachable[pairs[~v.masked[pairs[:, 1]], 0]] = True
    assert torch.equal(v.latent_targets, (v.masked & reachable) | ~v.masked)
    for name in ["displacements", "in_masked_structure", "gradient_targets"]:
        assert getattr(v, name) is None
    assert v.pairs is None and v.vectors is None


def test_extras_shapes_and_consistency(large_batch):
    systems, options, v = _views(large_batch, p_def=0.5, second_rotation=True)
    ho = v.extras["high_order"]
    n_structures = len(systems)
    n_atoms = sum(len(s) for s in systems)
    n_pairs = ho.pair_index.shape[0]
    n_centers = ho.centers.shape[0]
    dtype = systems[0].positions.dtype
    expected = {
        "branch_masked": ((n_structures,), torch.bool),
        "branch_deformed": ((n_structures,), torch.bool),
        "rotations": ((n_atoms, 3, 3), dtype),
        "rotations2": ((n_atoms, 3, 3), dtype),
        "u_student": ((n_atoms, 3), dtype),
        "grad_smooth": ((n_atoms, 3, 3), dtype),
        "u_smooth": ((n_atoms, 3), dtype),
        "field_scale": ((n_structures,), dtype),
        "pair_index": ((n_pairs, 2), torch.long),
        "clean_vectors": ((n_pairs, 3), dtype),
        "observed_vectors": ((n_pairs, 3), dtype),
        "hidden_source": ((n_pairs,), torch.bool),
        "boundary": ((n_atoms,), torch.bool),
        "context": ((n_atoms,), torch.bool),
        "centers": ((n_centers,), torch.long),
        "center_amplitudes": ((n_centers, 9), dtype),
        "species": ((n_atoms,), torch.int32),
        "species_index": ((n_atoms,), torch.long),
        "structure_index": ((n_atoms,), torch.long),
        "student_pairs": ((ho.student_pairs.shape[0], 2), torch.long),
        "student_distances": ((ho.student_pairs.shape[0],), dtype),
        "teacher_cap_fraction": ((n_structures,), dtype),
    }
    assert {f.name for f in fields(HighOrderExtras)} == set(expected)
    for name, (shape, field_dtype) in expected.items():
        value = getattr(ho, name)
        assert value.shape == shape, name
        assert value.dtype == field_dtype, name
        assert not value.requires_grad, name

    # branches are complementary
    assert torch.equal(ho.branch_masked, ~ho.branch_deformed)
    # per-atom bookkeeping
    sizes = torch.tensor([len(s) for s in systems])
    assert torch.equal(
        ho.structure_index, torch.repeat_interleave(torch.arange(n_structures), sizes)
    )
    types = torch.cat([s.types for s in systems])
    assert torch.equal(ho.species, types)
    atomic_types = torch.tensor(large_batch[2]["atomic_types"])
    assert torch.equal(atomic_types[ho.species_index], types.long())
    # masked atoms only in the mask-only branch
    assert not (v.masked & ho.branch_deformed[ho.structure_index]).any()
    # pairs: clean, below the cutoff, within a structure
    assert (ho.clean_vectors.norm(dim=-1) < options.cutoff).all()
    assert torch.equal(
        ho.structure_index[ho.pair_index[:, 0]],
        ho.structure_index[ho.pair_index[:, 1]],
    )
    assert torch.equal(ho.hidden_source, v.masked[ho.pair_index[:, 1]])
    # boundary and context partition the visible atoms
    assert not (ho.boundary & ho.context).any()
    assert torch.equal(ho.boundary | ho.context, ~v.masked)
    touches = ho.hidden_source & ~v.masked[ho.pair_index[:, 0]]
    assert torch.equal(
        ho.boundary.nonzero().squeeze(-1),
        ho.pair_index[touches, 0].unique(),
    )
    # the smooth part is the field of the deformed structures
    deformed_atoms = ho.branch_deformed[ho.structure_index]
    assert not ho.u_smooth[~deformed_atoms].any()
    assert not ho.grad_smooth[~deformed_atoms].any()
    assert ho.u_smooth[deformed_atoms].abs().max() > 0
    # student pairs: the kept pairs of the student views, below the cutoff
    assert (ho.student_distances < options.cutoff).all()
    assert not v.masked[ho.student_pairs].any()
    offset, pairs, distances = 0, [], []
    for s, st in zip(systems, v.student, strict=True):
        nl = st.get_neighbor_list(options)
        d = nl.values.reshape(-1, 3).norm(dim=-1).detach()
        close = d < options.cutoff
        pairs.append(nl.samples.values[close, :2] + offset)
        distances.append(d[close])
        offset += len(s)
    assert torch.equal(ho.student_pairs, torch.cat(pairs))
    torch.testing.assert_close(ho.student_distances, torch.cat(distances))


def test_periodic_structures_use_the_periodic_field(large_batch, monkeypatch):
    calls = []
    smooth = ho_views.smooth_field_with_gradient
    periodic = ho_views.periodic_field_with_gradient

    def spy_smooth(*args, **kwargs):
        calls.append("smooth")
        return smooth(*args, **kwargs)

    def spy_periodic(*args, **kwargs):
        calls.append("periodic")
        return periodic(*args, **kwargs)

    monkeypatch.setattr(ho_views, "smooth_field_with_gradient", spy_smooth)
    monkeypatch.setattr(ho_views, "periodic_field_with_gradient", spy_periodic)
    _views(large_batch, p_def=1.0)
    assert calls == ["smooth", "periodic"]  # the molecule, then the crystal


def test_unknown_species_raises(small_batch):
    systems, options, hypers = small_batch
    hypers = copy.deepcopy(hypers)
    hypers["atomic_types"] = [1, 6]  # QM9 frames also hold N and O
    with pytest.raises(ValueError, match="atomic_types"):
        build_high_order_views(
            systems, options, hypers, torch.Generator().manual_seed(0), True
        )


def _devices():
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", _devices())
def test_views_follow_the_device(large_batch, device):
    """Same random draws on every device: the views agree with the CPU ones."""
    systems, options, hypers = large_batch
    systems = [s.to(dtype=torch.float32) for s in systems]
    _, _, ref = _views((systems, options, hypers), p_def=0.5, second_rotation=True)
    moved = [s.to(device=device) for s in systems]
    _, _, v = _views((moved, options, hypers), p_def=0.5, second_rotation=True)
    ho, ref_ho = v.extras["high_order"], ref.extras["high_order"]
    for f in fields(HighOrderExtras):
        value = getattr(ho, f.name)
        assert value.device.type == device.type, f.name
        torch.testing.assert_close(
            value.cpu(), getattr(ref_ho, f.name), atol=1e-5, rtol=1e-5
        )
    for name in ["student", "teacher"]:
        for s, r in zip(getattr(v, name), getattr(ref, name), strict=True):
            assert s.positions.device.type == device.type
            torch.testing.assert_close(s.positions.detach().cpu(), r.positions)


def test_student_displacement_is_smooth_plus_bounded_iid():
    """``u_S = u_smooth + iid`` exactly, with the iid part centred and capped and the
    smooth part scaled as a whole (never capped per atom)."""
    generator = torch.Generator().manual_seed(0)
    field = 0.4 * torch.randn(30, 3, dtype=torch.float64, generator=generator)
    gradient = torch.randn(30, 3, 3, dtype=torch.float64, generator=generator)
    iid = 0.05 * torch.randn(30, 3, dtype=torch.float64, generator=generator)
    max_norm, sigma = 0.25, 0.05
    smooth = field - field.mean(0)
    rms = 0.4
    u_smooth, grad_smooth, u_student, field_scale = ho_views._student_displacement(
        field, gradient, iid, sigma, max_norm, rms
    )
    bound = min(3 * sigma, max_norm / 2)
    expected_iid = ho_views.cap_norm(iid - iid.mean(0), bound)
    assert torch.equal(u_student, u_smooth + expected_iid)
    normalise = rms / smooth.pow(2).sum(-1).mean().sqrt()
    scale = (max_norm - bound) / (normalise * smooth.norm(dim=-1).max())
    assert scale < 1  # the bound acts on this field
    torch.testing.assert_close(field_scale, scale)
    torch.testing.assert_close(u_smooth, scale * normalise * smooth)
    torch.testing.assert_close(grad_smooth, scale * normalise * gradient)
    assert u_student.norm(dim=-1).max() <= max_norm + 1e-12
    # a small field is left unscaled (by the safety factor), at the requested RMS
    u_small, g_small, _, s_small = ho_views._student_displacement(
        0.01 * field, gradient, iid, sigma, max_norm, 0.01
    )
    assert float(s_small) == 1.0
    r_small = (0.01 * smooth).pow(2).sum(-1).mean().sqrt()
    torch.testing.assert_close(
        u_small.pow(2).sum(-1).mean().sqrt(), torch.tensor(0.01, dtype=torch.float64)
    )
    torch.testing.assert_close(u_small, 0.01 * (0.01 * smooth) / r_small)
    torch.testing.assert_close(g_small, 0.01 * gradient / r_small)


def test_student_displacement_of_a_zero_field_is_finite():
    u_smooth, grad, u_student, scale = ho_views._student_displacement(
        torch.zeros(1, 3), torch.zeros(1, 3, 3), torch.zeros(1, 3), 0.03, 0.25, 0.05
    )
    assert torch.isfinite(u_student).all() and torch.isfinite(grad).all()
    assert float(scale) == 1.0


@pytest.mark.parametrize("n_atoms", [1, 2])
def test_student_displacement_without_centred_displacement_has_no_gradient(n_atoms):
    """A unit-amplitude field that is uniform over the atoms up to rounding (the
    views pass it centred) has no centred displacement to rescale to the requested
    RMS: its smooth part and gradient are zero, not the raw gradient inflated by
    ``rms / 0``."""
    field = 1e-9 * torch.randn(n_atoms, 3, dtype=torch.float64)
    gradient = torch.randn(n_atoms, 3, 3, dtype=torch.float64)
    u_smooth, grad, u_student, scale = ho_views._student_displacement(
        field, gradient, torch.zeros(n_atoms, 3, dtype=torch.float64), 0.03, 0.25, 0.05
    )
    assert torch.equal(u_smooth, torch.zeros_like(u_smooth))
    assert torch.equal(grad, torch.zeros_like(grad))
    assert float(scale) == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_structures_without_smooth_displacement_have_zero_jet_targets(dtype):
    """Structures of MAD on whose atoms the smooth field is uniform, on the
    deform-only branch: a monomer, a one-atom periodic cell and a CsCl-type cell
    (a reciprocal-lattice wave with even index sum takes the same value on both
    atoms; all six do with probability 1/64). No smooth displacement, so a zero
    smooth-field gradient (it was the raw gradient times ``rms / ~1e-15``, ~1e11)."""
    frames = [
        ase.Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[2.5] * 3, pbc=True),
    ]
    for seed in range(4):
        _, _, v = _views(make_batch(frames, dtype), seed=seed, p_def=1.0)
        ho = v.extras["high_order"]
        assert bool(ho.branch_deformed.all())
        assert torch.equal(ho.grad_smooth, torch.zeros_like(ho.grad_smooth))
        assert torch.equal(ho.u_smooth, torch.zeros_like(ho.u_smooth))
    # CsCl: the seeds whose six waves all have an even index sum
    cscl = ase.Atoms(
        "CO", scaled_positions=[[0, 0, 0], [0.5] * 3], cell=[3.1] * 3, pbc=True
    )
    uniform = 0
    for seed in range(400):
        _, _, v = _views(make_batch([cscl], dtype), seed=seed, p_def=1.0)
        ho = v.extras["high_order"]
        if bool(ho.branch_deformed.all()) and float(ho.u_smooth.abs().max()) == 0.0:
            uniform += 1
            assert torch.equal(ho.grad_smooth, torch.zeros_like(ho.grad_smooth))
    assert uniform > 0


@pytest.mark.parametrize("rms", [0.02, 0.05])
def test_smooth_field_has_the_requested_rms(large_batch, rms):
    """The smooth field of every deformed structure (molecule and periodic cell) has
    a per-structure RMS displacement ``field_amplitude`` before the safety scaling,
    i.e. ``RMS(u_smooth) = field_scale * field_amplitude``."""
    _, _, v = _views(large_batch, p_def=1.0, field_amplitude=rms, iid_sigma=0.0)
    ho = v.extras["high_order"]
    assert ho.field_scale.shape == (2,)
    for k in range(2):
        u = ho.u_smooth[ho.structure_index == k]
        assert torch.allclose(u.mean(0), torch.zeros(3, dtype=u.dtype), atol=1e-12)
        measured = u.pow(2).sum(-1).mean().sqrt()
        torch.testing.assert_close(measured, ho.field_scale[k] * rms)
        assert ho.field_scale[k] == 1.0


def test_field_scale_is_one_without_deformation(large_batch):
    _, _, v = _views(large_batch, p_def=0.0)
    assert (v.extras["high_order"].field_scale == 1.0).all()


def test_safety_scale_binds_for_a_large_amplitude(large_batch):
    _, _, v = _views(large_batch, p_def=1.0, field_amplitude=0.5)
    ho = v.extras["high_order"]
    assert (ho.field_scale < 1.0).all()
    assert (
        ho.u_student.norm(dim=-1).max() <= large_batch[2]["noise"]["skin"] / 2 + 1e-12
    )


def test_default_safety_scale_is_one_for_most_structures():
    """With the default amplitude the safety scaling is inactive for the large
    majority of QM9 molecules and of ~1000-atom periodic cells."""
    hypers_batch = make_batch(ase.io.read(QM9_PATH, ":100"), torch.float64)
    _, _, v = _views(hypers_batch, p_def=1.0)
    qm9 = v.extras["high_order"].field_scale
    assert float((qm9 == 1.0).double().mean()) >= 0.9
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(10):
        grid = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing="ij"), -1)
        pos = 1.6 * grid.reshape(-1, 3) + 0.1 * rng.standard_normal((1000, 3))
        frames.append(ase.Atoms("C1000", positions=pos, cell=[16.0] * 3, pbc=True))
    big = make_batch(frames, torch.float64)
    _, _, v = _views(big, p_def=1.0)
    periodic = v.extras["high_order"].field_scale
    assert float((periodic == 1.0).double().mean()) >= 0.9


def test_deformed_views_split_smooth_and_iid(large_batch):
    _, _, v = _views(large_batch, p_def=1.0, iid_sigma=0.0)
    ho = v.extras["high_order"]
    assert torch.equal(ho.u_student, ho.u_smooth)
    _, _, v = _views(large_batch, p_def=1.0)
    ho = v.extras["high_order"]
    hypers = large_batch[2]
    max_norm = hypers["noise"]["skin"] / 2
    bound = min(3 * hypers["high_order"]["iid_sigma"], max_norm / 2)
    assert (ho.u_student - ho.u_smooth).norm(dim=-1).max() <= bound + 1e-12
    assert ho.u_student.norm(dim=-1).max() <= max_norm + 1e-12


@pytest.mark.parametrize("periodic", [False, True])
def test_grad_smooth_is_the_gradient_of_u_smooth(large_batch, monkeypatch, periodic):
    """Finite differences of the stored (scaled) smooth field, regenerated from the
    same random state, match ``grad_smooth`` at every atom."""
    name = "periodic_field_with_gradient" if periodic else "smooth_field_with_gradient"
    original = getattr(ho_views, name)
    recorded = []

    def spy(positions, *args):
        recorded.append((positions.clone(), args[:-1], args[-1].get_state()))
        return original(positions, *args)

    monkeypatch.setattr(ho_views, name, spy)
    systems, _, v = _views(large_batch, p_def=1.0)
    ho = v.extras["high_order"]
    ((positions, args, state),) = recorded
    atoms = ho.structure_index == (1 if periodic else 0)

    def field(x):
        generator = torch.Generator()
        generator.set_state(state)
        return original(x, *args, generator)

    u0, g0 = field(positions)
    if not periodic:
        # the stored field has its rigid rotation removed (a constant gradient,
        # which the comparison below, relative to the mean gradient, ignores)
        u0, _ = ho_views._remove_rigid_rotation(positions, u0, g0)
    scale = ho.u_smooth[atoms].norm() / (u0 - u0.mean(0)).norm()
    torch.testing.assert_close(ho.u_smooth[atoms], scale * (u0 - u0.mean(0)))
    h = 1e-5
    numeric = torch.zeros(len(positions), 3, 3, dtype=torch.float64)
    for b in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[b] = h
        # every atom moves: the subtracted mean moves too
        up, um = field(positions + e)[0], field(positions - e)[0]
        numeric[:, :, b] = scale * (up - um) / (2 * h)
    grad = ho.grad_smooth[atoms]
    torch.testing.assert_close(numeric, grad - grad.mean(0), atol=1e-7, rtol=1e-6)


def test_partially_periodic_zero_cell_rows_raise(small_batch):
    systems, options, hypers = small_batch
    s = systems[0]
    pbc = torch.tensor([True, True, False])
    cell = torch.diag(torch.tensor([10.0, 10.0, 0.0]))
    broken = System(positions=s.positions, types=s.types, cell=cell, pbc=pbc)
    broken.add_neighbor_list(options, s.get_neighbor_list(options))
    with pytest.raises(ValueError, match="non-zero cell vectors"):
        build_high_order_views(
            [broken], options, hypers, torch.Generator().manual_seed(0), True
        )


def test_check_hypers_requires_a_skin():
    from metatrain.experimental.ssl_pretraining.trainer import check_hypers
    from metatrain.utils.architectures import get_default_hypers

    hypers = get_default_hypers("experimental.ssl_pretraining")["training"]
    hypers["views"]["builder"] = "high_order"
    hypers["noise"]["probability"] = 0.0
    hypers["noise"]["skin"] = 0.0
    with pytest.raises(ValueError, match="high_order"):
        check_hypers(hypers, cutoff=4.0)


def _rigid_rotation_fit(positions, u):
    """Least-squares angular velocity w of ``u ~ w x (r - centroid)``."""
    r = positions - positions.mean(dim=0)
    inertia = (r * r).sum() * torch.eye(3, dtype=r.dtype) - r.T @ r
    return torch.linalg.pinv(inertia) @ torch.cross(r, u, dim=-1).sum(dim=0)


def test_rigid_rotation_is_projected_out():
    """The projected field has no rigid-rotation part, and the projected gradient is
    its exact gradient (finite differences, the removed rotation held fixed)."""
    generator = torch.Generator().manual_seed(3)
    positions = 3.0 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    args = (0.15, 8.0, 6)
    state = generator.get_state()

    def field(x):
        g = torch.Generator()
        g.set_state(state)
        return smooth_field_with_gradient(x, *args, g)

    u, grad = field(positions)
    assert _rigid_rotation_fit(positions, u).norm() > 1e-4
    u_proj, grad_proj = ho_views._remove_rigid_rotation(positions, u, grad)
    assert _rigid_rotation_fit(positions, u_proj).norm() < 1e-12

    # the removed part is w x (r - c), with the constant gradient [w]_x
    removed = grad - grad_proj
    torch.testing.assert_close(removed, removed[:1].expand_as(removed))
    torch.testing.assert_close(removed[0], -removed[0].T)
    center = positions.mean(dim=0)
    torch.testing.assert_close(u - u_proj, (positions - center) @ removed[0].T)

    h = 1e-5
    numeric = torch.zeros_like(grad)
    for b in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[b] = h
        # every atom moves (the field's subtracted mean moves too); c is fixed
        up = field(positions + e)[0] - (positions + e - center) @ removed[0].T
        um = field(positions - e)[0] - (positions - e - center) @ removed[0].T
        numeric[:, :, b] = (up - um) / (2 * h)
    # the uniform shift also moves the field's mean: minus the mean raw gradient
    torch.testing.assert_close(numeric, grad_proj - grad.mean(0), atol=1e-7, rtol=1e-6)


def test_degenerate_structures_keep_a_finite_field():
    """A single atom and a collinear dimer: no (or an axial) rigid rotation."""
    for positions in (
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 1.2]], dtype=torch.float64),
    ):
        u = torch.randn(len(positions), 3, dtype=torch.float64)
        grad = torch.randn(len(positions), 3, 3, dtype=torch.float64)
        u_proj, grad_proj = ho_views._remove_rigid_rotation(positions, u, grad)
        assert torch.isfinite(u_proj).all() and torch.isfinite(grad_proj).all()
        assert _rigid_rotation_fit(positions, u_proj).norm() < 1e-10


def test_student_smooth_field_has_no_rigid_rotation(large_batch):
    systems, _, v = _views(large_batch, p_def=1.0)
    ho = v.extras["high_order"]
    molecule = ho.structure_index == 0
    positions = systems[0].positions.detach()
    assert _rigid_rotation_fit(positions, ho.u_smooth[molecule]).norm() < 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_near_linear_structures_keep_their_gradient_scale(seed):
    """A nearly linear molecule (QM9 frame 24, C2N2, and a synthetic 3-atom chain
    bent by 1e-3 Å) loses only the perpendicular rotations: its unidentifiable
    axial rotation is not fitted, so the gradient is not inflated."""
    chain = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 1e-3, 0.0], [2.4, 0.0, 0.0]], dtype=torch.float64
    )
    c2n2 = torch.tensor(ase.io.read(QM9_PATH, 24).positions, dtype=torch.float64)
    for positions in (chain, c2n2):
        generator = torch.Generator().manual_seed(seed)
        u, grad = smooth_field_with_gradient(positions, 0.15, 8.0, 6, generator)
        u_proj, grad_proj = ho_views._remove_rigid_rotation(positions, u, grad)
        assert grad_proj.abs().max() <= 2 * grad.abs().max()
        assert u_proj.abs().max() <= 2 * u.abs().max()


@pytest.mark.parametrize("p_def", [0.0, 1.0])
def test_second_rotation_rotates_the_same_student_input(large_batch, p_def):
    """``rot2`` is the student view (same mask, same displacement) under an extra
    rotation per structure, and ``rotations2`` is that total rotation per atom."""
    systems, options, off = _views(large_batch, p_def=p_def)
    _, _, on = _views(large_batch, p_def=p_def, second_rotation=True)
    assert set(on.extras) == {"high_order", "teacher_views", "student_views"}
    assert not set(on.extras) & set(RESERVED_EXTRAS)
    assert set(off.extras) == {"high_order", "teacher_views"}
    assert off.extras["high_order"].rotations2 is None
    # the first views and the bookkeeping do not depend on the option
    for f in fields(HighOrderExtras):
        if f.name != "rotations2":
            a = getattr(off.extras["high_order"], f.name)
            assert torch.equal(a, getattr(on.extras["high_order"], f.name)), f.name
    for a, b in zip(off.student, on.student, strict=True):
        assert torch.equal(a.positions, b.positions)

    ho = on.extras["high_order"]
    (second,) = on.extras["student_views"].values()
    assert list(on.extras["student_views"]) == ["rot2"]
    eye = torch.eye(3, dtype=ho.rotations2.dtype)
    assert torch.allclose(ho.rotations2 @ ho.rotations2.transpose(1, 2), eye)
    offset = 0
    for s, first, view in zip(systems, on.student, second, strict=True):
        n = len(s)
        g = ho.rotations[offset : offset + n]
        g2 = ho.rotations2[offset : offset + n]
        # the extra rotation is one rotation per structure
        extra = g2 @ g.transpose(1, 2)
        assert torch.allclose(extra, extra[0].expand(n, 3, 3), atol=1e-10)
        x = s.positions.detach() + ho.u_student[offset : offset + n]
        assert torch.allclose(view.positions, (x[:, None] @ g2.transpose(1, 2))[:, 0])
        assert torch.allclose(view.cell, s.cell @ g2[0].T)
        # the same pairs (masked pairs removed), rotated
        nl, nl2 = first.get_neighbor_list(options), view.get_neighbor_list(options)
        assert torch.equal(nl.samples.values, nl2.samples.values)
        torch.testing.assert_close(
            nl2.values.reshape(-1, 3), nl.values.reshape(-1, 3) @ extra[0].T
        )
        offset += n
    assert not torch.allclose(ho.rotations2, ho.rotations)


@pytest.mark.parametrize("position_gradients", [True, False])
def test_second_rotation_honours_position_gradients(small_batch, position_gradients):
    _, _, v = _views(
        small_batch, position_gradients=position_gradients, second_rotation=True
    )
    for system in v.extras["student_views"]["rot2"]:
        assert system.positions.requires_grad == position_gradients


def test_extra_student_views_are_encoded_with_gradients(
    step_data_two_rotations, step_data
):
    tokens = step_data_two_rotations.extras["student_tokens"]
    (name,) = step_data_two_rotations.extras["student_tokens_extra"]
    extra = step_data_two_rotations.extras["student_tokens_extra"][name]
    assert name == "rot2"
    assert extra.node.shape == tokens.node.shape
    assert extra.node.requires_grad and extra.edge.requires_grad
    assert "student_tokens_extra" in RESERVED_EXTRAS
    assert step_data.extras["student_tokens_extra"] == {}


def test_teacher_cap_fraction(large_batch):
    n_structures = len(large_batch[0])
    _, _, small = _views(large_batch, mode_amplitude=1e-4)
    ho = small.extras["high_order"]
    assert ho.teacher_cap_fraction.shape == (n_structures,)
    assert (ho.teacher_cap_fraction == 0).all()
    # a large amplitude makes the skin cap bind, and the fraction is a fraction
    _, _, large = _views(large_batch, mode_amplitude=5.0)
    fraction = large.extras["high_order"].teacher_cap_fraction
    assert ((fraction > 0) & (fraction <= 1)).all()
