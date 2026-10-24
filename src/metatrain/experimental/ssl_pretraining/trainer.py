import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from metatomic.torch import NeighborListOptions

from metatrain.pet.trainer import get_scheduler
from metatrain.utils.abc import TrainerInterface, common_upgrade_checkpoint
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    build_train_dataloaders,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.slurm import resolve_distributed
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.transfer import batch_to
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .backbones import BACKBONES, Backbone
from .conditioners import build_conditioners
from .objective import VIEW_BUILDERS, grad_jepa_loss
from .predictor import GradJEPAPredictor
from .terms import TermContext, build_terms
from .views import get_skin_neighbor_list_transform


def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, tau: float) -> None:
    """``teacher = tau * teacher + (1 - tau) * student``; buffers are copied.

    :param teacher: The EMA teacher, updated in place.
    :param student: The student.
    :param tau: The EMA momentum.
    """
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
            t.lerp_(s, 1.0 - tau)
        for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
            t.copy_(s)


class Trainer(TrainerInterface):
    __checkpoint_version__ = 2

    def __init__(self, hypers: Dict[str, Any]) -> None:
        super().__init__(hypers)
        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.step = 0
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.teacher_state_dict: Optional[Dict[str, Any]] = None
        # ``None`` when no active term uses the predictor
        self.predictor_state_dict: Optional[Dict[str, Any]] = None
        self.terms_state_dict: Optional[Dict[str, Any]] = None
        # set from the ``backbone`` model hyperparameter in ``setup``
        self.backbone: Backbone = BACKBONES["pet"]

    def setup(
        self, model_hypers: Dict[str, Any], dataset_info: DatasetInfo
    ) -> MetatrainModel:
        if len(dataset_info.targets) > 0:
            raise ValueError(
                "ssl_pretraining is label-free: set `targets: {}` in the "
                f"dataset options (got targets {list(dataset_info.targets)})"
            )
        # plain PET model hypers (without the ``backbone`` key) select PET
        self.backbone = BACKBONES[model_hypers.get("backbone", "pet")]
        self.backbone.validate(model_hypers)
        return self.backbone.setup(model_hypers, dataset_info)

    def restart(
        self,
        model: MetatrainModel,
        dataset_info: DatasetInfo,
        model_hypers: Dict[str, Any],
    ) -> MetatrainModel:
        raise ValueError(
            "ssl_pretraining runs are resumed with the `restart_from` trainer option, "
            "not with `mtt train --restart`"
        )

    def train(
        self,
        model: MetatrainModel,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in type(model.core).__supported_dtypes__
        if resolve_distributed(None):
            raise ValueError(
                "ssl_pretraining does not support distributed training yet"
            )
        device = devices[0]
        # the view builders also read the atomic types of the dataset (a runtime
        # key, not a hyperparameter: ``self.hypers`` is unchanged)
        hypers = {**self.hypers, "atomic_types": list(model.dataset_info.atomic_types)}

        student = model.core
        if hypers["restart_from"] is not None:
            self._load_restart_state(hypers["restart_from"], model)
        options = self.backbone.neighbor_list_options(student)
        check_hypers(hypers, options.cutoff)
        context = TermContext(
            dims=self.backbone.token_dims(student),
            cutoff=options.cutoff,
            atomic_types=list(model.dataset_info.atomic_types),
            num_probes=hypers["num_probes"],
            rotation_group=hypers["rotation_group"],
            feature_dim=self.backbone.feature_dim(student),
        )
        terms = build_terms(hypers["objective"]["terms"], context)
        ROOT_LOGGER.info("Active loss terms: " + ", ".join(term.name for term in terms))
        # the terms with parameters or state, by name
        term_modules = torch.nn.ModuleDict(
            {term.name: term for term in terms if isinstance(term, torch.nn.Module)}
        )
        # the predictor exists only if an active term reads its prediction
        predictor: Optional[GradJEPAPredictor] = None
        if any(getattr(term, "uses_prediction", False) for term in terms):
            predictor = GradJEPAPredictor(
                feature_dim=self.backbone.feature_dim(student),
                hidden_dim=hypers["predictor"]["hidden_dim"],
                probe_dim=hypers["predictor"]["probe_dim"],
                num_rbf=hypers["predictor"]["num_rbf"],
                cutoff=options.cutoff,
                atomic_types=model.dataset_info.atomic_types,
                species_query=hypers["predictor"]["species_query"],
                num_layers=hypers["predictor"]["num_layers"],
                conditioners=build_conditioners(hypers["predictor"]["conditioners"]),
            )
        teacher = copy.deepcopy(student)
        if self.teacher_state_dict is not None:
            teacher.load_state_dict(self.teacher_state_dict)
        if predictor is not None and self.predictor_state_dict is not None:
            predictor.load_state_dict(self.predictor_state_dict)
        if self.terms_state_dict is not None:
            term_modules.load_state_dict(self.terms_state_dict)
        # ``copy.deepcopy`` and ``load_state_dict`` leave metatensor Labels buffers
        # (PET's ``single_label``) on the CPU: every copy and state load comes
        # first, then everything is moved to the device
        for module in [model, teacher, predictor, term_modules]:
            if module is not None:
                module.to(device=device, dtype=dtype)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        collate_fn_train, collate_fn_val = get_collate_fns(options, hypers)
        if hypers["num_workers"] is None:
            num_workers = get_num_workers()
        else:
            num_workers = hypers["num_workers"]
            validate_num_workers(num_workers)
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=[None] * len(train_datasets),
            collate_fn_train=collate_fn_train,
            batch_size=hypers["batch_size"],
            max_atoms_per_batch=hypers["max_atoms_per_batch"],
            min_atoms_per_batch=hypers["min_atoms_per_batch"],
            num_workers=num_workers,
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)
        val_dataloader = CombinedDataLoader(
            build_val_dataloaders(
                val_datasets=val_datasets,
                val_distributed_samplers=[None] * len(val_datasets),
                collate_fn_val=collate_fn_val,
                batch_size=hypers["batch_size"],
                max_atoms_per_batch=hypers["max_atoms_per_batch"],
                num_workers=num_workers,
            ),
            shuffle=False,
        )
        if len(val_dataloader) == 0:
            raise ValueError("ssl_pretraining needs a non-empty validation set")

        # student, predictor, terms: the order of the saved optimizer state
        if hypers["freeze_backbone"]:
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            parameters: List[torch.nn.Parameter] = []
        else:
            parameters = list(student.parameters())
        if predictor is not None:
            parameters += list(predictor.parameters())
        parameters += list(term_modules.parameters())
        if len(parameters) == 0:
            raise ValueError(
                "`freeze_backbone: true` needs a predictor or a term with parameters "
                "to train"
            )
        optimizer = torch.optim.AdamW(
            parameters, lr=hypers["learning_rate"], weight_decay=hypers["weight_decay"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)
        lr_scheduler = get_scheduler(optimizer, hypers, len(train_dataloader))
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        total_steps = hypers["num_epochs"] * len(train_dataloader)
        tau_start, tau_end = hypers["ema_tau_range"]
        # offset by the step so that a resumed run does not replay the same views
        generator = torch.Generator().manual_seed(hypers["seed"] + self.step)
        start_epoch = 0 if self.epoch is None else self.epoch + 1
        if self.best_metric is None:
            self.best_metric = float("inf")
        epoch = start_epoch
        metric_logger: Optional[MetricLogger] = None

        for epoch in range(start_epoch, hypers["num_epochs"]):
            for sampler in epoch_samplers:
                sampler.set_epoch(epoch)
            student.train()
            term_modules.train()
            train_metrics: Dict[str, float] = {}
            n_train = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                systems, _, _ = unpack_batch(batch)
                systems, _, _ = batch_to(systems, {}, {}, dtype=dtype, device=device)
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    predictor,
                    terms,
                    systems,
                    options,
                    hypers,
                    generator,
                    backbone=self.backbone,
                    step_fraction=self.step / max(1, total_steps),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, hypers["grad_clip_norm"])
                optimizer.step()
                lr_scheduler.step()
                tau = tau_start + (tau_end - tau_start) * min(
                    1.0, self.step / max(1, total_steps)
                )
                if not hypers["freeze_backbone"]:
                    # a frozen backbone: the teacher stays equal to the student
                    ema_update(teacher, student, tau)
                self.step += 1
                _accumulate(train_metrics, {"loss": float(loss.detach()), **metrics})
                n_train += 1

            student.eval()
            term_modules.eval()
            val_metrics: Dict[str, float] = {}
            n_val = 0
            for index, batch in enumerate(val_dataloader):
                systems, _, _ = unpack_batch(batch)
                systems, _, _ = batch_to(systems, {}, {}, dtype=dtype, device=device)
                val_generator = torch.Generator().manual_seed(hypers["seed"] + index)
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    predictor,
                    terms,
                    systems,
                    options,
                    hypers,
                    val_generator,
                    with_rank=True,
                    backbone=self.backbone,
                    step_fraction=1.0,
                )
                _accumulate(val_metrics, {"loss": float(loss.detach()), **metrics})
                n_val += 1

            train_info = {k: v / max(1, n_train) for k, v in train_metrics.items()}
            val_info = {k: v / max(1, n_val) for k, v in val_metrics.items()}
            # per-epoch term state (e.g. the order curriculum of
            # ``rotation_conditioned_latent``), before it is checkpointed
            for term in terms:
                hook = getattr(term, "on_validation_epoch_end", None)
                if hook is not None:
                    hook(val_info)
            if metric_logger is None:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=model.dataset_info,
                    initial_metrics=[train_info, val_info],
                    names=["training", "validation"],
                )
            if epoch % hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[train_info, val_info],
                    epoch=epoch,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            # the weights handed to fine-tuning and export (``best_model_state_dict``)
            if hypers["export_weights"] == "final_teacher":
                self.best_metric = val_info["loss"]
                self.best_epoch = epoch
                self.best_model_state_dict = _with_core(model, teacher)
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            elif -val_info["rankme"] < self.best_metric:  # "best_rankme_student"
                self.best_metric = -val_info["rankme"]
                self.best_epoch = epoch
                self.best_model_state_dict = copy.deepcopy(model.state_dict())
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            self.epoch = epoch
            self.optimizer_state_dict = optimizer.state_dict()
            self.scheduler_state_dict = lr_scheduler.state_dict()
            self.teacher_state_dict = teacher.state_dict()
            self.predictor_state_dict = (
                None if predictor is None else predictor.state_dict()
            )
            self.terms_state_dict = term_modules.state_dict()
            is_last = epoch == hypers["num_epochs"] - 1
            if epoch % hypers["checkpoint_interval"] == 0 or is_last:
                self.save_checkpoint(
                    model, Path(checkpoint_dir) / f"model_{epoch}.ckpt"
                )

    def _load_restart_state(self, path: str, model: MetatrainModel) -> None:
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        if "grad_jepa" not in checkpoint:
            raise ValueError(f"'{path}' is not an ssl_pretraining checkpoint")
        version = checkpoint.get("trainer_ckpt_version")
        if (
            not isinstance(version, int)
            or not 1 <= version <= self.__checkpoint_version__
        ):
            raise ValueError(
                f"cannot resume from '{path}': its ssl_pretraining trainer checkpoint "
                f"version {version} is not supported (this trainer reads versions 1 "
                f"to {self.__checkpoint_version__})"
            )
        checkpoint = self.upgrade_checkpoint(checkpoint)
        previous = model_from_checkpoint(checkpoint, context="restart")
        model.load_state_dict(previous.state_dict())
        state = checkpoint["grad_jepa"]
        self.epoch = checkpoint["epoch"]
        self.step = state["step"]
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        self.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        self.best_epoch = checkpoint["best_epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.best_model_state_dict = MetatrainModel.get_state_dict_from_checkpoint(
            checkpoint, "best_model_state_dict"
        )
        self.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]
        self.teacher_state_dict = state["teacher_state_dict"]
        self.predictor_state_dict = state["predictor_state_dict"]
        self.terms_state_dict = state["terms_state_dict"]

    def save_checkpoint(self, model: MetatrainModel, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint(self.best_model_state_dict)
        checkpoint["core"]["epoch"] = self.epoch
        checkpoint["core"]["best_epoch"] = self.best_epoch
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
                # the pre-rename key, kept so that old checkpoints stay readable
                "grad_jepa": {
                    "step": self.step,
                    "teacher_state_dict": self.teacher_state_dict,
                    "predictor_state_dict": self.predictor_state_dict,
                    "terms_state_dict": self.terms_state_dict,
                },
            }
        )
        torch.save(checkpoint, check_file_extension(path, ".ckpt"))

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: Dict[str, Any],
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        raise ValueError(
            "ssl_pretraining runs are resumed with the `restart_from` trainer option"
        )

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: Dict[str, Any], version: Optional[int] = None
    ) -> Dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "trainer", checkpoints
        )


def _with_core(model: MetatrainModel, core: torch.nn.Module) -> Dict[str, Any]:
    """A copy of ``model.state_dict()`` with the core weights taken from ``core``.

    :param model: The model whose state dict is copied.
    :param core: A module with the same parameters as ``model.core``.
    :return: The new state dict.
    """
    state = copy.deepcopy(model.state_dict())
    for name, value in core.state_dict().items():
        # ``_extra_state`` entries (PET checkpoint v18+) are not tensors
        if isinstance(value, torch.Tensor):
            state[f"core.{name}"] = value.detach().clone()
        else:
            state[f"core.{name}"] = copy.deepcopy(value)
    return state


def _accumulate(total: Dict[str, float], values: Dict[str, float]) -> None:
    for name, value in values.items():
        total[name] = total.get(name, 0.0) + value


def get_collate_fns(
    options: NeighborListOptions, hypers: Dict[str, Any]
) -> Tuple[CollateFn, CollateFn]:
    """Training and validation collate functions.

    Both compute the skin neighbor list. With ``global_augmentation``, training
    batches are first randomly rotated as a whole (both views) by metatrain's
    ``O3Augmenter``, as in PET training, so that the teacher does not always see
    the dataset frame. Validation batches are not augmented.

    :param options: PET neighbor list options.
    :param hypers: The trainer hypers.
    :return: The training and validation collate functions.
    """
    skin_transform = get_skin_neighbor_list_transform(options, hypers["noise"]["skin"])
    train_callables: List[Callable[..., Any]] = []
    if hypers["global_augmentation"]:
        augmenter = O3Augmenter(target_info_dict={}, extra_data_info_dict={})
        train_callables.append(augmenter.apply_random_augmentations)
    train_callables.append(skin_transform)
    return (
        CollateFn(target_keys=[], callables=train_callables),
        CollateFn(target_keys=[], callables=[skin_transform]),
    )


def check_hypers(hypers: Dict[str, Any], cutoff: float) -> None:
    """Validate the hypers that depend on each other or on the PET cutoff.

    :param hypers: The trainer hypers.
    :param cutoff: The PET cutoff.
    :raises ValueError: If the hypers are inconsistent.
    """
    mask = hypers["mask"]
    for name in ["fraction_range", "ball_radius_range", "ema_tau_range"]:
        values = hypers[name] if name == "ema_tau_range" else mask[name]
        if len(values) != 2 or values[0] > values[1]:
            raise ValueError(f"`{name}` must be [low, high] with low <= high")
    if not 0.0 < mask["fraction_range"][1] < 1.0:
        raise ValueError("`mask.fraction_range` must lie in (0, 1)")
    if mask["ball_radius_range"][1] > cutoff:
        raise ValueError(
            f"`mask.ball_radius_range` must not exceed the PET cutoff ({cutoff})"
        )
    if mask["topology_cutoff"] > cutoff:
        raise ValueError(
            f"`mask.topology_cutoff` must not exceed the PET cutoff ({cutoff})"
        )
    if len(mask["ego_hops"]) == 0 or min(mask["ego_hops"]) < 1:
        raise ValueError("`mask.ego_hops` must be a non-empty list of positive ints")
    if hypers["views"]["builder"] not in VIEW_BUILDERS:
        raise ValueError(
            f"unknown view builder '{hypers['views']['builder']}' (registered: "
            f"{', '.join(VIEW_BUILDERS)})"
        )
    if hypers["num_probes"] < 1:
        raise ValueError("`num_probes` must be at least 1")
    if hypers["noise"]["probability"] > 0 and hypers["noise"]["skin"] <= 0:
        raise ValueError("`noise.skin` must be positive when noise is enabled")
    if hypers["views"]["builder"] == "high_order" and hypers["noise"]["skin"] <= 0:
        raise ValueError("`noise.skin` must be positive with the high_order views")
    check_term_combination(hypers)


# the terms that keep the representation from collapsing
ANTI_COLLAPSE_TERMS = ("vicreg", "sigreg", "vicreg_averaged")
# the terms that read the Grad-JEPA views (latent and gradient targets)
GRAD_JEPA_VIEW_TERMS = ("latent", "gradient", "gradient_spread")
# the terms that read the extras of the high-order views
HIGH_ORDER_VIEW_TERMS = (
    "residual_multipole",
    "deformation_jets",
    "chirality",
    "equivariant_vicreg",
    "vicreg_averaged",
    "rotation_conditioned_latent",
)
# the terms that read the second student rotation of the high-order views
SECOND_ROTATION_TERMS = ("equivariant_vicreg", "vicreg_averaged")


def check_term_combination(hypers: Dict[str, Any]) -> None:
    """Validate the active loss terms against each other and against the views.

    :param hypers: The trainer hypers.
    :raises ValueError: If the active terms do not fit together or with the views.
    """
    terms = {k: v for k, v in hypers["objective"]["terms"].items() if v is not None}
    builder = hypers["views"]["builder"]
    for name in terms:
        if name in HIGH_ORDER_VIEW_TERMS and builder != "high_order":
            raise ValueError(f"the '{name}' term needs `views.builder: high_order`")
        if name in GRAD_JEPA_VIEW_TERMS and builder == "high_order":
            raise ValueError(
                f"the '{name}' term needs the Grad-JEPA views (`views.builder: "
                "grad_jepa`), not the high_order views"
            )
        if (
            name in SECOND_ROTATION_TERMS
            and not hypers["high_order"]["second_rotation"]
        ):
            raise ValueError(
                f"the '{name}' term needs `high_order.second_rotation: true`"
            )
    anti_collapse = [name for name in ANTI_COLLAPSE_TERMS if name in terms]
    if len(anti_collapse) > 1 or (
        len(anti_collapse) == 0 and not hypers["allow_no_anti_collapse"]
    ):
        raise ValueError(
            f"`objective.terms` must hold exactly one anti-collapse term among "
            f"{', '.join(ANTI_COLLAPSE_TERMS)} (got {len(anti_collapse)}); "
            "`allow_no_anti_collapse: true` allows none (design pilots only)"
        )
    for name in SECOND_ROTATION_TERMS:
        if name in terms and hypers["rotation_group"] == "none":
            raise ValueError(
                f"the '{name}' term needs random student rotations "
                "(`rotation_group` is 'none')"
            )
    if "equivariant_vicreg" in terms:
        group = terms["equivariant_vicreg"]["rotation_group"]
        if group != hypers["rotation_group"]:
            raise ValueError(
                f"the 'equivariant_vicreg' term's `rotation_group` ({group}) must "
                f"equal the trainer's `rotation_group` ({hypers['rotation_group']})"
            )
