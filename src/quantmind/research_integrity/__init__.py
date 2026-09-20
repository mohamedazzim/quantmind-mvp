from .acquisition_gate import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    false_pass_gate_passes,
    positive_control_gate_passes,
)
from .trial_ledger import ResearchBudget, TrialBudgetExceeded, TrialContext, TrialLedger
from .harness import ResearchHarness, ResearchTask, derive_experiment_id, derive_strategy_id
from .holdout import (
    FinalEvaluationResult,
    HoldoutManager,
    HoldoutSecurityError,
    HoldoutState,
)
from .artifacts import (
    ArtifactImmutabilityError,
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactStatus,
    ArtifactType,
    load_net_returns_array,
    load_oos_returns_verified,
    load_signal_timestamps_array,
    write_oos_returns_artifact,
)
from .population import (
    DsrPopulationInputs,
    Eict1PairwiseInput,
    Eict1PopulationInputs,
    EictCorr1InputBuilder,
    EligibleTrial,
    ProductionPopulationQuery,
    ReturnDistributionMetadata,
    TrialProvenance,
    align_pair,
    build_dsr_inputs,
    compute_return_distribution,
)
from .statistical_validation import (
    DeflatedSharpeCalculator,
    DeflatedSharpeResult,
    EictClusterResult,
    EictCorr1Calculator,
    StatisticalValidationPipeline,
    compute_population_hash,
)

__all__ = [
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "false_pass_gate_passes",
    "positive_control_gate_passes",
    "ResearchBudget",
    "TrialBudgetExceeded",
    "TrialContext",
    "TrialLedger",
    "ResearchHarness",
    "ResearchTask",
    "derive_experiment_id",
    "derive_strategy_id",
    "FinalEvaluationResult",
    "HoldoutManager",
    "HoldoutSecurityError",
    "HoldoutState",
    # artifacts
    "ArtifactImmutabilityError",
    "ArtifactRecord",
    "ArtifactRegistry",
    "ArtifactStatus",
    "ArtifactType",
    "load_net_returns_array",
    "load_oos_returns_verified",
    "load_signal_timestamps_array",
    "write_oos_returns_artifact",
    # population
    "DsrPopulationInputs",
    "Eict1PairwiseInput",
    "Eict1PopulationInputs",
    "EictCorr1InputBuilder",
    "EligibleTrial",
    "ProductionPopulationQuery",
    "ReturnDistributionMetadata",
    "TrialProvenance",
    "align_pair",
    "build_dsr_inputs",
    "compute_return_distribution",
    # statistical validation
    "DeflatedSharpeCalculator",
    "DeflatedSharpeResult",
    "EictClusterResult",
    "EictCorr1Calculator",
    "StatisticalValidationPipeline",
    "compute_population_hash",
]
