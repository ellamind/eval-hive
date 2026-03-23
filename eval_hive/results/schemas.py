"""Pydantic schemas for evaluation result data."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EvalConfig(BaseModel):
    """Per-result-file environment and harness metadata.

    Stored as a nested struct — rarely queried directly but useful for
    reproducibility and debugging.
    """

    lm_eval_version: Optional[str] = Field(
        None, description='lm-eval-harness version, e.g. "0.4.10"'
    )
    git_hash: Optional[str] = Field(None, description="lm-eval-harness git commit")
    transformers_version: Optional[str] = Field(None, description="transformers library version")
    model_name: Optional[str] = Field(
        None, description="Model identifier as known by lm-eval (HF path or local path)"
    )
    model_source: Optional[str] = Field(
        None, description='How the model was loaded, e.g. "local-completions", "hf"'
    )
    max_length: Optional[int] = Field(None, description="Max sequence length used")
    total_eval_time_seconds: Optional[float] = Field(
        None, description="Wall-clock time for the full evaluation run"
    )
    tokenizer_pad_token: Optional[str] = Field(None, description="Pad token string")
    tokenizer_eos_token: Optional[str] = Field(None, description="EOS token string")
    tokenizer_bos_token: Optional[str] = Field(None, description="BOS token string")
    eot_token_id: Optional[int] = Field(None, description="End-of-text token ID")
    chat_template: Optional[str] = Field(None, description="Chat template if used")
    chat_template_sha: Optional[str] = Field(None, description="SHA hash of the chat template")
    system_instruction: Optional[str] = Field(None, description="System instruction if used")
    system_instruction_sha: Optional[str] = Field(
        None, description="SHA hash of the system instruction"
    )


class TaskConfig(BaseModel):
    """Per-task configuration details.

    Stored as a nested struct — useful for auditing exactly how a task
    was prompted and scored.
    """

    task_version: Optional[str] = Field(
        None, description="Task definition version from lm-eval-harness"
    )
    task_hash: Optional[str] = Field(
        None, description="Hash fingerprint of the task definition"
    )
    dataset_path: Optional[str] = Field(
        None, description='HuggingFace dataset path, e.g. "ellamind/arc-german-preview"'
    )
    output_type: Optional[str] = Field(
        None, description='Raw lm-eval-harness output type, e.g. "loglikelihood", "generate_until"'
    )
    dataset_name: Optional[str] = Field(
        None, description='HuggingFace dataset subset name, e.g. "ARC-Challenge"'
    )
    repeats: Optional[int] = Field(None, description="Number of repeats per sample")
    doc_to_text: Optional[str] = Field(None, description="Jinja template or literal for prompt")
    doc_to_target: Optional[str] = Field(None, description="Jinja template or literal for target")
    doc_to_choice: Optional[str] = Field(
        None, description="Jinja template or literal list for choices (serialized as string)"
    )
    generation_kwargs: Optional[str] = Field(
        None,
        description="Generation parameters as JSON string (max_gen_toks, temperature, etc.)",
    )
    should_decontaminate: Optional[bool] = Field(None, description="Decontamination flag")


class ScoreRow(BaseModel):
    """One row in scores.parquet.

    Composite key: (model, step, task, metric, metric_filter)
    """

    # Score identity (composite key)
    model: str = Field(..., description='Model name, e.g. "qwen3-1.7b"')
    step: Optional[int] = Field(
        None, description="Training step; None for models where not available"
    )
    task: str = Field(..., description='Task name, e.g. "arc_challenge"')
    metric: str = Field(..., description='Metric name, e.g. "acc_norm"')
    metric_filter: str = Field(..., description='Metric filter, e.g. "none"')

    # Score values
    score: Optional[float] = Field(None, description="Primary score value")
    score_stderr: Optional[float] = Field(None, description="Standard error of score")

    # Task metadata (denormalized from tasks)
    task_type: Literal["eval_suite", "task_group", "benchmark"] = Field(
        ..., description="Level in the task hierarchy"
    )
    parent_task: Optional[str] = Field(
        None, description="Parent task in hierarchy; null at root"
    )
    task_display_name: Optional[str] = Field(
        None, description="Human-friendly task name"
    )
    num_fewshot: Optional[int] = Field(None, description="Number of few-shot examples")
    task_formulation: Optional[
        Literal[
            "multiple_choice",
            "rank_choice",
            "generative",
            "bits_per_byte",
        ]
    ] = Field(
        None,
        description="task formulation",
    )
    higher_is_better: Optional[bool] = Field(
        None, description="Whether a higher score is better for this metric"
    )
    language: Optional[str] = Field(
        None, description='ISO 639-3 language code, e.g. "deu", "eng"'
    )
    n_samples: Optional[int] = Field(
        None, description="Number of samples in the task"
    )
    subtask_tree: Optional[dict[str, list[str]]] = Field(
        None,
        description=(
            "Adjacency map of the subtask hierarchy that composes this aggregated "
            "score.  Keys are group/suite names mapping to their immediate children; "
            "leaf benchmarks only appear as list values, never as keys.  Serialized "
            "as a JSON string in parquet.  Populated for eval_suite and task_group "
            "rows; None for leaf benchmarks."
        ),
    )

    # Training metadata
    train_batch_size: Optional[int] = Field(
        None,
        description="Batch size in tokens; allows computing tokens_trained = train_batch_size * step",
    )
    tokens_trained: Optional[int] = Field(
        None, description="Total tokens trained at this checkpoint"
    )

    # Provenance
    eval_date: Optional[str] = Field(
        None, description="ISO 8601 timestamp of when the evaluation was run"
    )
    result_source: Optional[str] = Field(
        None, description="Absolute path to the result JSON this row was parsed from"
    )

    # Nested metadata structs (rarely queried, full detail)
    eval_config: Optional[EvalConfig] = Field(
        None, description="Per-file eval environment metadata"
    )
    task_config: Optional[TaskConfig] = Field(
        None, description="Per-task configuration details"
    )

    # Model metadata
    model_path: Optional[str] = Field(
        None, description="Path to model weights (local path or HuggingFace model ID)"
    )
    is_checkpoint: bool = Field(
        ..., description="Whether this is an intermediate checkpoint or a final model"
    )
    model_display_name: str = Field(..., description="Human-friendly model name")

    @model_validator(mode="after")
    def _derive_training_metadata(self) -> ScoreRow:
        """Derive missing training metadata when possible.

        * ``tokens_trained = train_batch_size * step`` (when both are known)
        * ``train_batch_size = tokens_trained // step`` (when both are known)
        """
        if (
            self.tokens_trained is None
            and self.train_batch_size is not None
            and self.step is not None
        ):
            self.tokens_trained = self.train_batch_size * self.step
        if (
            self.train_batch_size is None
            and self.tokens_trained is not None
            and self.step is not None
            and self.step > 0
        ):
            self.train_batch_size = self.tokens_trained // self.step
        return self
