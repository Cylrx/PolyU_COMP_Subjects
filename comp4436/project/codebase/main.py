"""COMP4436 AIoT Streaming Simulation System.

Two-phase workflow:
  Phase 1: Model/Profile setup
  Phase 2: Experiments
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from config import (
    ArrivalConfig,
    ModelVariantBlacklistConfig,
    PlotConfig,
    ProfilingConfig,
    SimulationConfig,
)
from evaluation.metrics import MetricsSummary
from core.compat import check_environment
from core.console import (
    console,
    error,
    print_environment_status,
    print_experiment_results,
    print_model_variants,
    print_profile_info,
    status,
    success,
)
from core.types import LENET_MNIST_PROFILES, RESNET_CIFAR10_PROFILES, ProfileCache
from experiments import admission, dispatch
from experiments.runner import ExperimentRunner

from rich.prompt import IntPrompt

CACHE_DIR = Path("cache")
WEIGHTS_DIR = CACHE_DIR / "weights"
PROFILES_DIR = CACHE_DIR / "profiles"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


# -- Phase 1: Setup ----------------------------------------------------------


def _discover_profiles() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(PROFILES_DIR.glob("*.json"))


def _profile_kind_label(profile: ProfileCache) -> str:
    return "mock" if profile.is_mock else "profiled"


def _profile_menu_label(path: Path, profile: ProfileCache) -> str:
    metadata = profile.metadata
    return (
        f"{path.stem} "
        f"[{_profile_kind_label(profile)}] "
        f"({metadata.n_samples} samples, "
        f"{len(metadata.model_variants)} variants, "
        f"edge={metadata.edge_device}, cloud={metadata.cloud_device})"
    )


def _generated_profile_path(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROFILES_DIR / f"{prefix}_{timestamp}.json"


def _save_profile(profile: ProfileCache, path: Path) -> ProfileCache:
    profile.save(path)
    success(f"Profile saved to {path}")
    return profile


def _select_pipeline() -> str:
    """Ask user to choose the model pipeline. Returns 'resnet' or 'lenet'."""
    console.print()
    console.print("  Select model pipeline:")
    console.print("    [cyan][1][/] ResNet-152 + CIFAR-10")
    console.print("    [cyan][2][/] LeNet-5 + MNIST")
    choice = IntPrompt.ask("    Select", default=1, choices=["1", "2"])
    return "resnet" if choice == 1 else "lenet"


_PIPELINE_PREFIX = {"resnet": "resnet_cifar10", "lenet": "lenet_mnist"}


def _train_and_profile_new(pipeline: str) -> ProfileCache | None:
    console.print()
    console.rule("[bold]Pretrain + Preprofile New")
    profiling_cfg = ProfilingConfig()

    env = check_environment()
    print_environment_status(env)

    if not env.can_profile:
        error("Cannot train a new model — PyTorch and torchvision are required.")
        return None
    if not env.cuda_available:
        error("Cannot create a new dual-device profile — CUDA is not available.")
        return None

    from models.compression import default_compressions
    from models.profiler import build_profile

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if pipeline == "resnet":
        from models.resnet import (
            CIFAR10_INPUT_SHAPE,
            RESNET_ARCHITECTURE,
            RESNET_DISPLAY_NAME,
            build_cifar10_datasets,
            fine_tune_resnet,
            train_resnet,
            save_weights,
        )

        model, accuracy = train_resnet(device=env.device, data_dir=str(DATA_DIR))
        input_shape = CIFAR10_INPUT_SHAPE
        train_dataset, test_dataset = build_cifar10_datasets(str(DATA_DIR))
        weight_path = WEIGHTS_DIR / f"{RESNET_ARCHITECTURE}_cifar10_{timestamp}.pt"
        save_weights(model, weight_path, {
            "accuracy": accuracy, "architecture": RESNET_ARCHITECTURE,
            "dataset": "cifar10", "epochs": 50,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        description = f"{RESNET_DISPLAY_NAME} CIFAR-10 dual-device profile ({timestamp})"
        fine_tune_variant = partial(fine_tune_resnet, device=env.device)
    else:
        from models.lenet import (
            DEFAULT_WIDTH,
            build_mnist_datasets,
            fine_tune_lenet,
            save_weights,
            train_lenet,
        )

        model, accuracy = train_lenet(
            width=DEFAULT_WIDTH, device=env.device, data_dir=str(DATA_DIR),
        )
        input_shape = (1, 28, 28)
        train_dataset, test_dataset = build_mnist_datasets(str(DATA_DIR))
        weight_path = WEIGHTS_DIR / f"lenet_mnist_{timestamp}.pt"
        save_weights(model, weight_path, {
            "accuracy": accuracy, "width": DEFAULT_WIDTH,
            "architecture": "lenet5", "dataset": "mnist", "epochs": 15,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        description = f"LeNet-5 MNIST dual-device profile ({timestamp})"
        fine_tune_variant = partial(fine_tune_lenet, device=env.device)

    profile = build_profile(
        base_model=model,
        compressions=default_compressions(input_shape),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        input_shape=input_shape,
        fine_tune_variant=fine_tune_variant,
        edge_device="cpu",
        cloud_device="cuda",
        description=description,
        edge_latency_scale_factor=profiling_cfg.edge_latency_scale_factor,
        cloud_latency_scale_factor=profiling_cfg.cloud_latency_scale_factor,
    )
    prefix = _PIPELINE_PREFIX[pipeline]
    saved = _save_profile(profile, _generated_profile_path(f"{prefix}_profile"))
    print_profile_info(saved.metadata)
    return saved


def _generate_mock_profile(pipeline: str) -> ProfileCache:
    console.print()
    console.rule("[bold]Generate Mock Profile")
    profiling_cfg = ProfilingConfig()

    profiles = RESNET_CIFAR10_PROFILES if pipeline == "resnet" else LENET_MNIST_PROFILES
    profile = ProfileCache.mock(
        n_samples=1000,
        model_profiles=profiles,
        seed=42,
        edge_latency_scale_factor=profiling_cfg.edge_latency_scale_factor,
        cloud_latency_scale_factor=profiling_cfg.cloud_latency_scale_factor,
    )
    prefix = _PIPELINE_PREFIX[pipeline]
    saved = _save_profile(profile, _generated_profile_path(f"{prefix}_mock"))
    print_profile_info(saved.metadata)
    return saved


def _select_existing_profile(auto_select_first: bool) -> ProfileCache | None:
    found = _discover_profiles()
    if not found:
        error("No saved profiles found.")
        return None

    options: list[tuple[Path, ProfileCache | None, str]] = []

    for path in found:
        try:
            loaded = ProfileCache.load(path)
            options.append((path, loaded, _profile_menu_label(path, loaded)))
        except Exception as e:
            options.append((path, None, f"{path.stem} [invalid] ({e})"))

    console.print()
    console.rule("[bold]Use Existing Profile")

    for i, (_, _, label) in enumerate(options, 1):
        console.print(f"  [cyan][{i}][/] {label}")

    if auto_select_first:
        for i, (_, cache, label) in enumerate(options, 1):
            if cache is not None:
                status(f"Auto-selecting: {label}")
                selection = i
                break
        else:
            error("No valid saved profiles found.")
            return None
    else:
        selection = IntPrompt.ask(
            "  Select",
            default=1,
            choices=[str(i) for i in range(1, len(options) + 1)],
        )

    path, profile, label = options[selection - 1]
    if profile is None:
        error("Selected profile is invalid. Please choose a valid profile.")
        return None

    success(f"Loaded profile: {path.name}")
    print_profile_info(profile.metadata)
    return profile


def _prepare_profile(args: argparse.Namespace) -> ProfileCache:
    auto_choice: int | None = None
    if args.use_cache:
        auto_choice = 2
    elif args.retrain or args.reprofile:
        auto_choice = 1

    while True:
        console.print()
        console.rule("[bold]Phase 1: Model/Profile Setup")
        console.print("  [cyan][1][/] Pretrain + Preprofile new")
        console.print("  [cyan][2][/] Use existing profile [bold](Recommended)[/]")

        if auto_choice is not None:
            choice = auto_choice
            auto_choice = None
            status(f"Auto-selecting setup option [{choice}]")
        else:
            choice = IntPrompt.ask(
                "  Select",
                default=2 if _discover_profiles() else 1,
                choices=["1", "2"],
            )

        if choice == 1:
            pipeline = _select_pipeline()
            profile = _train_and_profile_new(pipeline)
            if profile is not None:
                return profile
            continue

        if choice == 2:
            profile = _select_existing_profile(auto_select_first=args.use_cache)
            args.use_cache = False
            if profile is not None:
                return profile
            continue


# -- Phase 2: Experiments ----------------------------------------------------

SUITES = [
    ("dispatch", dispatch),
    ("admission", admission),
]


def _run_experiments(profile: ProfileCache) -> None:
    console.print()
    console.rule("[bold]Phase 2: Running Experiments")

    sim_cfg = SimulationConfig()
    plot_cfg = PlotConfig()
    arrival_cfg = ArrivalConfig()
    runner = ExperimentRunner(
        profile=profile,
        available_models=profile.available_models,
        metric_interval=plot_cfg.metric_interval,
    )

    for idx, (suite_name, suite_module) in enumerate(SUITES, 1):
        console.print(f"[bold]Suite {str(idx)}: {suite_name}")
        if sim_cfg.n_repetitions > 1:
            status(f"Averaging over {sim_cfg.n_repetitions} repetitions")

        accumulated: dict[str, list[MetricsSummary]] = {}
        for rep in range(sim_cfg.n_repetitions):
            rep_sim = replace(sim_cfg, seed=sim_cfg.seed + rep)
            rep_arr = replace(arrival_cfg, seed=arrival_cfg.seed + rep)
            results = runner.run_all(suite_module.build(profile, rep_sim, rep_arr))
            for name, summary in results.items():
                accumulated.setdefault(name, []).append(summary)

        averaged = {
            name: MetricsSummary.average(sums)
            for name, sums in accumulated.items()
        }
        cols = getattr(suite_module, "console_columns", lambda: None)()
        print_experiment_results(averaged, columns=cols)

        console.print()
        status("Generating reports...")
        suite_module.report(
            averaged, OUTPUT_DIR,
            plot_config=plot_cfg,
            arrival_cfg=arrival_cfg,
            per_run=accumulated,
        )


def _apply_model_variant_blacklist(profile: ProfileCache) -> ProfileCache:
    blacklist_cfg = ModelVariantBlacklistConfig()
    if not blacklist_cfg.enabled:
        return profile
    return profile.blacklisted(blacklist_cfg.disabled_variants)


# -- Entry point --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="COMP4436 AIoT Simulation")
    parser.add_argument("--use-cache", action="store_true", help="Auto-select the first saved profile")
    parser.add_argument("--retrain", action="store_true", help="Auto-select pretrain + preprofile")
    parser.add_argument("--reprofile", action="store_true", help="Alias for --retrain")
    args = parser.parse_args()

    console.print()
    console.print("[bold blue]COMP4436 AIoT Streaming Simulation System[/]")
    console.print("[dim]Unified setup flow with pluggable strategies[/]")

    profile = _apply_model_variant_blacklist(_prepare_profile(args))
    print_model_variants(profile.available_models)
    _run_experiments(profile)

    console.print()
    success("All experiments complete.")


if __name__ == "__main__":
    main()
