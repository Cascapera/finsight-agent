"""Testes do contrato do golden set — puros (sem banco, sem LLM, sem rede)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from finsight.evals.dataset import SEED_DATASET, EvalDataset, EvalSample


def test_seed_dataset_is_internally_consistent() -> None:
    """
    O seed precisa ser coerente: cada ground_truth deve ser derivável dos seus
    ground_truth_contexts. Não dá para verificar semântica sem LLM, mas
    garantimos o mínimo estrutural — todo caso tem pergunta, gabarito e ao menos
    um contexto de referência (senão context_recall/precision ficam sem âncora).
    """
    assert len(SEED_DATASET) > 0
    for sample in SEED_DATASET.samples:
        assert sample.question.strip()
        assert sample.ground_truth.strip()
        assert sample.ground_truth_contexts, f"{sample.id} sem ground_truth_contexts"


def test_strict_mode_rejects_wrong_types() -> None:
    """
    strict=True deve recusar coerção silenciosa. Um ground_truth_contexts que
    não é lista é erro de autoria do golden set — queremos falha explícita.
    """
    with pytest.raises(ValidationError):
        EvalSample(
            id="bad",
            question="q",
            ground_truth="a",
            ground_truth_contexts="não é lista",  # type: ignore[arg-type]
        )


def test_filter_by_ticker() -> None:
    ds = EvalDataset(
        samples=[
            EvalSample(id="a", question="q", ground_truth="g", ticker="PNOR3"),
            EvalSample(id="b", question="q", ground_truth="g", ticker="OUTRO4"),
        ]
    )
    filtered = ds.filter_by_ticker("PNOR3")
    assert len(filtered) == 1
    assert filtered.samples[0].id == "a"


def test_json_round_trip(tmp_path: Path) -> None:
    """Salvar e recarregar não pode perder nem deformar nenhum caso."""
    path = tmp_path / "golden.json"
    SEED_DATASET.to_json(path)
    reloaded = EvalDataset.from_json(path)
    assert reloaded == SEED_DATASET


def test_from_json_accepts_bare_list(tmp_path: Path) -> None:
    """O loader tolera o arquivo ser uma lista direta (sem a chave 'samples')."""
    path = tmp_path / "list.json"
    path.write_text(
        '[{"id": "x", "question": "q", "ground_truth": "g", '
        '"ground_truth_contexts": ["c"], "ticker": null}]',
        encoding="utf-8",
    )
    ds = EvalDataset.from_json(path)
    assert len(ds) == 1
    assert ds.samples[0].id == "x"
