"""Grad-JEPA predictor.

One round of local message passing from context atoms to target atoms. Output
vectors are built only from input vectors (pair directions, student gradients,
displacements) times scalar coefficients, so the gradient prediction is
rotation-covariant by construction even though PET is not. It is also linear in
the probe vector ``v``, so that matching random slices matches the Jacobian.
"""

from typing import Dict, Optional

import torch


def _mlp(d_in: int, d_hidden: int, d_out: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, d_hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(d_hidden, d_hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(d_hidden, d_out),
    )


def _cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(torch.pi * distances / cutoff) + 1.0) * (distances < cutoff)


class GradJEPAPredictor(torch.nn.Module):
    """Predicts teacher latents and teacher random-scalar gradients at target atoms.

    :param feature_dim: Dimension ``D`` of the encoder per-atom features.
    :param hidden_dim: Width of the message MLPs.
    :param probe_dim: Rank of the learned probe projection (coefficients that
        multiply pair directions and displacements are linear in ``U v``).
    :param num_rbf: Number of radial basis functions.
    :param cutoff: Cutoff radius of the message passing.
    :param atomic_types: Atomic types, for the optional species query.
    :param species_query: If true, masked atoms are queried with their species.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        probe_dim: int,
        num_rbf: int,
        cutoff: float,
        atomic_types: list[int],
        species_query: bool,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.probe_dim = probe_dim
        self.species_query = species_query
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_width = cutoff / num_rbf
        self.register_buffer(
            "type_to_index",
            torch.full((max(atomic_types) + 1,), -1, dtype=torch.long),
        )
        for index, atomic_type in enumerate(atomic_types):
            self.type_to_index[atomic_type] = index
        self.species_embedding = torch.nn.Embedding(len(atomic_types), hidden_dim)

        self.query_context = torch.nn.Linear(feature_dim, hidden_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.source = torch.nn.Linear(feature_dim, hidden_dim)
        # message inputs: query, source, rbf(d), |delta_i|, |delta_j - delta_i|,
        # rhat . (delta_j - delta_i)
        self.message = _mlp(2 * hidden_dim + num_rbf + 3, hidden_dim, hidden_dim)
        self.probe_projection = torch.nn.Linear(feature_dim, probe_dim, bias=False)
        # coefficients: rhat (probe_dim), delta_j - delta_i (probe_dim), g_j (1)
        self.edge_coefficients = torch.nn.Linear(hidden_dim, 2 * probe_dim + 1)
        # self coefficients for context targets: g_i (1), delta_i (probe_dim)
        self.self_coefficients = _mlp(hidden_dim, hidden_dim, probe_dim + 1)
        self.latent_head = _mlp(2 * hidden_dim, hidden_dim, feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        gradients: Optional[torch.Tensor],
        probes: torch.Tensor,
        displacements: torch.Tensor,
        types: torch.Tensor,
        masked: torch.Tensor,
        pairs: torch.Tensor,
        vectors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        :param features: Student features ``(n_atoms, D)`` (rows of masked atoms
            are ignored).
        :param gradients: Student gradients in the teacher frame
            ``(n_probes, n_atoms, 3)`` (zero at masked atoms), or ``None`` to
            predict only the latents.
        :param probes: Probe vectors ``(n_probes, D)``.
        :param displacements: Teacher-view displacements ``(n_atoms, 3)``.
        :param types: Atomic types ``(n_atoms,)``.
        :param masked: Boolean mask of masked atoms ``(n_atoms,)``.
        :param pairs: Clean full-structure pairs ``(n_pairs, 2)`` (center, neighbor),
            batch-global indices, restricted to distances below the cutoff.
        :param vectors: Pair vectors ``(n_pairs, 3)`` in the teacher frame.
        :return: ``{"latent": (n_atoms, D), "gradient": (n_probes, n_atoms, 3)}``,
            defined for every atom (callers select the target atoms); only
            ``latent`` when ``gradients`` is ``None``.
        """
        n_atoms = features.shape[0]
        context = ~masked

        query = torch.where(
            masked.unsqueeze(-1),
            self.mask_token.expand(n_atoms, -1),
            self.query_context(features),
        )
        if self.species_query:
            species = self.species_embedding(self.type_to_index[types])
            query = query + masked.unsqueeze(-1) * species

        # only messages from context neighbors
        keep = context[pairs[:, 1]]
        centers, neighbors = pairs[keep, 0], pairs[keep, 1]
        vectors = vectors[keep]
        distances = vectors.norm(dim=-1).clamp(min=1e-12)
        directions = vectors / distances.unsqueeze(-1)
        rbf = torch.exp(
            -(((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        )
        relative_delta = displacements[neighbors] - displacements[centers]
        scalars = torch.stack(
            [
                displacements[centers].norm(dim=-1),
                relative_delta.norm(dim=-1),
                (directions * relative_delta).sum(dim=-1),
            ],
            dim=-1,
        )
        messages = self.message(
            torch.cat(
                [query[centers], self.source(features[neighbors]), rbf, scalars], dim=-1
            )
        )
        weights = _cosine_cutoff(distances, self.cutoff).unsqueeze(-1)
        messages = messages * weights
        norm = torch.zeros(n_atoms, 1, dtype=features.dtype, device=features.device)
        norm = norm.index_add(0, centers, weights) + 1.0

        aggregated = torch.zeros_like(query).index_add(0, centers, messages) / norm
        latent = self.latent_head(torch.cat([query, aggregated], dim=-1))
        if gradients is None:
            return {"latent": latent}

        # gradient prediction, linear in the probes
        projected = self.probe_projection(probes)  # (n_probes, probe_dim)
        coefficients = self.edge_coefficients(messages)
        c_direction = coefficients[:, : self.probe_dim] @ projected.T  # (E, P)
        c_delta = coefficients[:, self.probe_dim : 2 * self.probe_dim] @ projected.T
        c_gradient = coefficients[:, -1:]  # (E, 1)
        edge_vectors = (
            c_direction.T.unsqueeze(-1) * directions
            + c_delta.T.unsqueeze(-1) * relative_delta
            + c_gradient.T.unsqueeze(-1) * gradients[:, neighbors]
        )  # (P, E, 3)
        gradient = torch.zeros_like(gradients).index_add(1, centers, edge_vectors)
        gradient = gradient / norm

        self_coefficients = self.self_coefficients(query) * context.unsqueeze(-1)
        self_gradient = self_coefficients[:, -1:].T.unsqueeze(-1) * gradients
        self_delta = (self_coefficients[:, :-1] @ projected.T).T.unsqueeze(-1)
        gradient = gradient + self_gradient + self_delta * displacements

        return {"latent": latent, "gradient": gradient}
