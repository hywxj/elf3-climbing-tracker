"""Training control decisions driven by completed episodes and evaluation."""

from dataclasses import dataclass, field


@dataclass
class EvaluationGuard:
    patience: int = 3
    low_success: float = 0.2
    established_success: float = 0.9
    best_start: float = 0.0
    best_tail: float = 0.0
    bad_streak: int = 0
    stalled_checks: int = 0
    phase: float | None = None
    failures: list = field(default_factory=list)

    def inspect(self, start_success, tail_success, phase, phase_success, training_success):
        regression = (
            self.best_start >= self.established_success and start_success < self.low_success
        ) or (self.best_tail >= self.established_success and tail_success < self.low_success)
        self.best_start = max(self.best_start, start_success)
        self.best_tail = max(self.best_tail, tail_success)
        self.bad_streak = self.bad_streak + 1 if regression else 0
        if phase != self.phase:
            self.stalled_checks = 0
            self.phase = phase
        stalled = phase_success < self.low_success and training_success < self.low_success
        self.stalled_checks = self.stalled_checks + 1 if stalled else 0
        reason = None
        if self.bad_streak >= self.patience:
            reason = "evaluation_regression"
        elif self.stalled_checks >= self.patience:
            reason = "no_task_success"
        if reason:
            self.failures.append(reason)
        return reason


def save_checkpoint_atomically(runner, path, infos):
    """Keep the previous assessed checkpoint intact if serialization is interrupted."""
    temporary = path.with_name("." + path.name + ".writing.pt")
    runner.save(str(temporary), infos=infos)
    temporary.replace(path)


def finalize_checkpoint(runner, output, completed, contract, interrupted=False):
    """Publish only weights paired with a completed evaluation; retain partial work separately."""
    import shutil

    if interrupted:
        save_checkpoint_atomically(
            runner,
            output / "interrupted.pt",
            infos={
                "evaluated": False,
                "last_evaluated_iteration": completed,
                "contract": contract,
            },
        )
    assessed = output / ("latest.pt" if completed else "best.pt")
    shutil.copyfile(assessed, output / "checkpoint.pt")
    return assessed
