import pytest
from pydantic import ValidationError

from ni43101.config import Settings
from ni43101.orchestrator import (
    NI43101Orchestrator,
    OrchestratorConfigurationError,
)


class _UnusedAgent:
    pass


@pytest.mark.parametrize("rounds", [0, 4])
def test_settings_enforces_at_most_three_business_rounds(rounds: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_revise_rounds=rounds)


@pytest.mark.parametrize("score", [7.99, 10.01])
def test_settings_enforces_safe_pass_score(score: float) -> None:
    with pytest.raises(ValidationError):
        Settings(pass_score=score)


@pytest.mark.parametrize("rounds", [0, 4])
def test_orchestrator_rejects_out_of_protocol_round_count(rounds: int) -> None:
    with pytest.raises(
        OrchestratorConfigurationError,
        match="between 1 and 3",
    ):
        NI43101Orchestrator(
            _UnusedAgent(),
            _UnusedAgent(),
            max_revise_rounds=rounds,
        )


def test_orchestrator_rejects_pass_threshold_below_eight() -> None:
    with pytest.raises(
        OrchestratorConfigurationError,
        match="between 8 and 10",
    ):
        NI43101Orchestrator(
            _UnusedAgent(),
            _UnusedAgent(),
            pass_score=7,
        )


@pytest.mark.parametrize("tolerance", [0.049, 0.051, 0.10])
def test_settings_keeps_evaluation_protocol_at_five_percent(
    tolerance: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(field_tolerance=tolerance)
