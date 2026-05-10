"""Local k-agents-style tune-up automation, no API key required."""
from .registry import Registry, Experiment, build_default_registry
from .policy import RulePolicy, PARAMS, Decision
from .orchestrator import (
    QubitOrchestrator,
    TuneUpResult,
    run_parallel_tuneups,
    summarise,
    to_records,
)
from .plots import (
    plot_pipeline_graph,
    plot_parameter_spread,
    plot_runtime_dashboard,
)
