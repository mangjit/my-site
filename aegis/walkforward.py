"""Chronological folds. The test year is not a training year. The holdout is not a test year."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    procedure: str
    train_years: tuple[int, ...]
    test_year: int
    holdout_years: tuple[int, ...]


def calendar_years(years: list[int]) -> list[int]:
    return sorted(set(years))


def build_folds(years_present: list[int], min_train_years: int, holdout_years: int) -> tuple[list[Fold], list[int]]:
    ordered = calendar_years(years_present)
    if holdout_years < 1:
        raise ValueError("A locked holdout of at least one year is required.")
    if len(ordered) <= min_train_years + holdout_years:
        raise ValueError(
            f"Need more than {min_train_years} training years plus {holdout_years} holdout year(s). "
            f"Found {ordered}."
        )
    holdout = ordered[-holdout_years:]
    usable = ordered[:-holdout_years]
    folds: list[Fold] = []
    for index in range(min_train_years, len(usable)):
        test_year = usable[index]
        expanding = tuple(usable[:index])
        rolling = tuple(usable[index - min_train_years : index])
        folds.append(Fold("expanding", expanding, test_year, tuple(holdout)))
        folds.append(Fold("rolling", rolling, test_year, tuple(holdout)))
    return folds, holdout


def purge_training_indices(
    candidate_indices: list[int],
    label_end_index: dict[int, int],
    test_start_index: int,
) -> tuple[list[int], list[int]]:
    """Drop training decisions whose label window touches the test period or later."""
    kept = []
    embargoed = []
    for index in candidate_indices:
        end = label_end_index.get(index)
        if end is None or end >= test_start_index:
            embargoed.append(index)
        else:
            kept.append(index)
    return kept, embargoed
