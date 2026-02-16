import re
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List


def _parse_human_int(value) -> int | None:
    """Parse a human-readable integer with optional suffix.

    Accepts plain integers or strings like ``"100B"``, ``"2T"``, ``"4M"``, ``"1.5B"``.

    Supported suffixes (case-insensitive): K (10^3), M (10^6), B (10^9), T (10^12).
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected int or string, got {type(value).__name__}")
    s = value.strip().replace("_", "")
    if not s:
        return None
    suffixes = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
    upper = s[-1].upper()
    if upper in suffixes:
        return int(float(s[:-1]) * suffixes[upper])
    return int(s)


class ModelEntry(BaseModel):
    """A single model (or checkpoint family) to evaluate."""

    path: str = Field(..., description="Local path or HuggingFace model ID")
    model_key: Optional[str] = Field(
        default=None,
        description=(
            "Model identifier used in the parquet 'model' column and output directory. "
            "Defaults to the config dict key. Set this to share a model key between "
            "a main model entry and its checkpoint entry (they differ only by step)."
        ),
    )
    checkpoint_pattern: Optional[str] = Field(
        default=None,
        description="Pattern for checkpoint subdirectories, e.g. 'checkpoint_{step}'",
    )
    steps: Optional[List[int]] = Field(
        default=None,
        description=(
            "List of checkpoint steps to evaluate. "
            "None means auto-discover all subdirectories matching checkpoint_pattern."
        ),
    )
    display_name: str = Field(
        ...,
        description="Human-friendly model name for result display.",
    )
    train_batch_size: Optional[int] = Field(
        default=None,
        description=(
            "Training batch size in tokens (e.g. 4194304). "
            "Recommended for checkpoint models. "
            "Used to compute tokens_trained = train_batch_size * step."
        ),
    )
    tokens_trained: Optional[int] = Field(
        default=None,
        description=(
            "Total tokens trained (e.g. 100000000000 or '100B'). "
            "Recommended for non-checkpoint models. "
            "For checkpoints, derived from train_batch_size * step."
        ),
    )

    @field_validator("tokens_trained", mode="before")
    @classmethod
    def parse_human_readable_int(cls, v):
        return _parse_human_int(v)

    @model_validator(mode="after")
    def validate_checkpoint_fields(self):
        if self.steps is not None and self.checkpoint_pattern is None:
            raise ValueError(
                "'steps' requires 'checkpoint_pattern' to be set"
            )
        if self.steps is not None and len(self.steps) == 0:
            raise ValueError("'steps' must not be empty when provided")
        return self

    @model_validator(mode="after")
    def warn_training_metadata(self):
        import warnings
        is_checkpoint = self.checkpoint_pattern is not None
        if is_checkpoint and self.train_batch_size is None:
            warnings.warn(
                f"Model '{self.display_name}': 'train_batch_size' is recommended "
                f"for checkpoint models (needed to compute tokens_trained per step)",
                stacklevel=2,
            )
        if not is_checkpoint and self.tokens_trained is None:
            warnings.warn(
                f"Model '{self.display_name}': 'tokens_trained' is recommended "
                f"for non-checkpoint models",
                stacklevel=2,
            )
        return self

    def _checkpoint_regex(self) -> re.Pattern[str]:
        """Compile checkpoint_pattern into a regex matching directory names."""
        regex_str = re.escape(self.checkpoint_pattern).replace(r"\{step\}", r"(\d+)")
        return re.compile(f"^{regex_str}$")

    def resolve_checkpoints(self) -> List[tuple[int, Path]]:
        """Return sorted (step, path) pairs for this model entry.

        Always discovers from disk. steps is an optional filter.

        - No checkpoint_pattern → empty list (single model, not a checkpoint series).
        - checkpoint_pattern set → glob directory, extract step numbers.
        - steps set → filter to only matching step numbers.
        """
        if self.checkpoint_pattern is None:
            return []

        base = Path(self.path)
        rx = self._checkpoint_regex()

        found = []
        if base.is_dir():
            for child in base.iterdir():
                if child.is_dir():
                    m = rx.match(child.name)
                    if m:
                        step = int(m.group(1))
                        if self.steps is None or step in self.steps:
                            found.append((step, child))
        return sorted(found)

    def resolve_model_paths(self) -> List[tuple[str, Path]]:
        """Return (label, path) pairs for all models/checkpoints to evaluate.

        Labels use the actual directory name (preserving zero-padding etc.):
        - Single model → [("main", path)]
        - Checkpoint series → [("checkpoint_0010000", path), ...]
        """
        checkpoints = self.resolve_checkpoints()
        if not checkpoints:
            return [("main", Path(self.path))]
        return [(ckpt_path.name, ckpt_path) for _step, ckpt_path in checkpoints]


# lm_eval CLI args that eval-hive constructs per-suite and must not be overridden
_RESERVED_LM_EVAL_ARGS = {
    "output_path",  # constructed by eval-hive per suite
    "model_args",   # built from eval.model_args
    "tasks",        # set per suite from suites_and_tasks
    "include_path", # set from eval_suite_path
}


class EvalSection(BaseModel):
    """The eval: section of the config."""

    eval_suite_path: Optional[str] = Field(
        default=None,
        description="Path to eval suite directory (lm-eval --include_path). None if only using built-in tasks.",
    )
    suites_and_tasks: List[str] = Field(
        ..., min_length=1, description="List of suite/task names to run"
    )
    model_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments joined as key=value,key=value for --model_args",
    )
    lm_eval_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Passthrough CLI flags for lm_eval (--key value)",
    )
    # Timeout and retry defaults for lm-eval API requests.
    # Injected into model_args unless already set there.
    timeout: int = Field(
        default=600,
        gt=0,
        description="HTTP request timeout in seconds for lm-eval API calls (default: 600)",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        description="Maximum retry attempts for failed lm-eval API requests (default: 5)",
    )

    @field_validator("lm_eval_args")
    @classmethod
    def validate_lm_eval_args(cls, v):
        """Reject lm_eval_args that eval-hive manages."""
        conflicts = set(v.keys()) & _RESERVED_LM_EVAL_ARGS
        if conflicts:
            raise ValueError(
                f"lm_eval_args contains keys managed by eval-hive: {sorted(conflicts)}. "
                f"Use the dedicated config fields instead."
            )
        return v


class EhConfig(BaseModel):
    """Top-level eval-hive configuration."""

    # SLURM Configuration
    job_name: str = Field(default="eval-hive", description="Name of the SLURM job")
    partition: str = Field(..., description="SLURM partition to use")
    account: str = Field(..., description="SLURM account to charge")
    qos: str = Field(..., description="Quality of Service for the job")
    cpus_per_node: int = Field(..., gt=0, description="Number of CPUs per node")
    gres_per_node: str = Field(
        ...,
        description="Generic resources per node (e.g. 'gpu:4')",
    )
    time_limit: str = Field(..., description="SLURM time limit")
    max_concurrent_jobs: int = Field(
        default=4,
        gt=0,
        description="Maximum number of concurrent SLURM array jobs",
    )
    additional_sbatch_args: Optional[Dict[str, str]] = Field(
        default=None, description="Additional SBATCH arguments as key-value pairs"
    )

    # Scaling Configuration
    num_inference_servers: int = Field(
        default=1, gt=0, description="Number of inference server instances per job"
    )
    num_nodes_per_inference_server: int = Field(
        default=1, gt=0, description="Number of nodes per inference server instance"
    )
    parallel_tasks: int = Field(
        default=1,
        ge=1,
        description=(
            "Max concurrent lm-eval processes per job (one per task). "
            "1 = sequential (current behavior)."
        ),
    )

    # Environment Configuration
    pixi_manifest: Optional[str] = Field(
        default=None, description="Path to the pixi manifest file"
    )
    pixi_env: Optional[str] = Field(
        default=None, description="Pixi environment name"
    )
    env_activation_command: Optional[str] = Field(
        default=None,
        description="Custom environment activation command (alternative to pixi)",
    )
    env_vars: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables as key-value pairs"
    )

    # Inference Server Configuration
    inference_server_command: Optional[str] = Field(
        default=None,
        description=(
            "Command to start the inference server. "
            "Use ${EH_PORT} and ${EH_MODEL_PATH} placeholders. "
            "Set to null for serverless backends (hf, nemo)."
        ),
    )

    # Health Check Configuration
    health_check_max_wait_minutes: int = Field(
        default=10, gt=0, description="Maximum time to wait for server health"
    )
    health_check_interval_seconds: int = Field(
        default=20, gt=0, description="Interval between health checks in seconds"
    )

    # Eval Configuration
    eval: EvalSection

    # Output Configuration
    output_path: Path = Field(..., description="Base path for result output")
    request_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for lm-eval request cache (overrides LM_HARNESS_CACHE_PATH). If not set, lm-eval uses its built-in default.",
    )

    # Models
    models: Dict[str, ModelEntry] = Field(
        ..., min_length=1, description="Models to evaluate"
    )

    # ── Validators ──────────────────────────────────────────────

    @field_validator("output_path", "request_cache_dir", mode="before")
    @classmethod
    def convert_paths(cls, v):
        if v is not None:
            return Path(v).absolute()
        return v

    @field_validator("inference_server_command", mode="before")
    @classmethod
    def strip_server_command(cls, v):
        if v is not None:
            return v.strip()
        return v

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, v):
        """Validate SLURM time limit format.

        Acceptable: "minutes", "minutes:seconds", "hours:minutes:seconds",
        "days-hours", "days-hours:minutes", "days-hours:minutes:seconds", or "0".
        """
        if not isinstance(v, str):
            raise ValueError("time_limit must be a string")

        if v == "0":
            return v

        if "-" in v:
            dash_parts = v.split("-")
            if len(dash_parts) != 2:
                raise ValueError(
                    "time_limit with days format must have exactly one dash"
                )
            days_part, hours_part = dash_parts
            try:
                int(days_part)
            except ValueError:
                raise ValueError("days part must be a number")
            colon_parts = hours_part.split(":")
            if len(colon_parts) not in (1, 2, 3):
                raise ValueError(
                    "hours part must be 'hours', 'hours:minutes', "
                    "or 'hours:minutes:seconds'"
                )
            for part in colon_parts:
                try:
                    int(part)
                except ValueError:
                    raise ValueError("all time components must be numbers")
        else:
            colon_parts = v.split(":")
            if len(colon_parts) not in (1, 2, 3):
                raise ValueError(
                    "time_limit must be 'minutes', 'minutes:seconds', "
                    "or 'hours:minutes:seconds'"
                )
            for part in colon_parts:
                try:
                    int(part)
                except ValueError:
                    raise ValueError("all time components must be numbers")

        return v

    @field_validator("additional_sbatch_args")
    @classmethod
    def validate_additional_sbatch_args(cls, v):
        """Reject SBATCH args that conflict with fields managed by eval-hive."""
        if v is None:
            return v

        reserved_args = {
            "job-name", "partition", "account", "qos", "array", "nodes",
            "cpus-per-task", "mem", "gres", "time", "signal", "output", "error",
            "dependency", "requeue", "no-requeue", "wait",
            "gpus", "gpus-per-node", "gpus-per-socket", "gpus-per-task",
            "gpu-bind", "gpu-freq",
            "mem-per-cpu", "mem-per-gpu", "mem-bind",
            "cpus-per-gpu", "cores-per-socket", "sockets-per-node",
            "threads-per-core",
            "ntasks", "ntasks-per-node", "ntasks-per-core",
            "ntasks-per-socket", "ntasks-per-gpu",
            "hint", "extra-node-info",
            "tres-bind", "tres-per-task",
        }

        for key in v:
            clean_key = key.lstrip("-")
            if clean_key in reserved_args:
                raise ValueError(
                    f"SBATCH argument '{key}' is reserved and cannot be overridden. "
                    f"Reserved args: {sorted(reserved_args)}"
                )
        return v

    @model_validator(mode="after")
    def validate_env_activation(self):
        """Ensure either pixi fields or env_activation_command is provided."""
        has_pixi = self.pixi_manifest is not None and self.pixi_env is not None
        has_custom = self.env_activation_command is not None

        if not has_pixi and not has_custom:
            raise ValueError(
                "Either 'pixi_manifest' and 'pixi_env' must both be set, "
                "or 'env_activation_command' must be set"
            )
        return self

    @model_validator(mode="after")
    def validate_server_scaling(self):
        """num_inference_servers > 1 requires an inference server command."""
        if self.num_inference_servers > 1 and self.inference_server_command is None:
            raise ValueError(
                "num_inference_servers > 1 requires 'inference_server_command' to be set "
                "(serverless backends cannot be load-balanced)"
            )
        return self


def load_config(config_path: str | Path) -> EhConfig:
    """Load and validate an eval-hive config from a YAML file."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return EhConfig(**data)
