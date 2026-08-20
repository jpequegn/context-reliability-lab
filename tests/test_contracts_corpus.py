from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_reliability_lab.contracts import Corpus, TaskFamily
from context_reliability_lab.corpus import generate_case, generate_corpus


def test_corpus_is_balanced_and_deterministic() -> None:
    first = generate_corpus()
    second = generate_corpus()
    assert len(first.cases) == 50
    assert Counter(case.family for case in first.cases) == {family: 10 for family in TaskFamily}
    assert first.digest == second.digest


def test_hidden_truth_is_separate_and_round_trips() -> None:
    corpus = generate_corpus()
    encoded = corpus.model_dump_json()
    restored = Corpus.model_validate_json(encoded)
    assert restored == corpus
    assert "hidden_truth" not in corpus.cases[0].model_dump()


def test_scope_canaries_are_never_in_visible_queries() -> None:
    corpus = generate_corpus()
    for case in corpus.cases:
        canary = corpus.hidden_truth[case.case_id].canary
        assert all(canary not in event.content for event in case.events)
        restricted = [memory for memory in case.initial_memory if memory.user_id != case.user_id]
        assert len(restricted) == 1
        assert canary in restricted[0].content


@given(st.sampled_from(list(TaskFamily)), st.integers(min_value=0, max_value=9))
def test_generated_cases_keep_positive_budgets(family: TaskFamily, index: int) -> None:
    case, truth = generate_case(family, index)
    assert case.token_budget > 0
    assert case.tool_budget >= 0
    assert truth.case_id == case.case_id


def test_invalid_truth_mapping_is_rejected() -> None:
    corpus = generate_corpus()
    with pytest.raises(ValueError, match="one-to-one"):
        Corpus(cases=corpus.cases, hidden_truth={})
