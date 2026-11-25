"""Study sample analysis tests against in-memory records."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

from config.study_config import load_study
from entropy import entropy_table
from samples import build_table, record_rows
from semantic_entropy import group_answers
from utilities.criteria import CRITERION_NAMES


def record(model, index, satisfied):
    criterion = CRITERION_NAMES[0]
    return {
        "paper_id": "123",
        "provider": model,
        "sample_index": index,
        "criteria_results": [{
            "criterion": criterion,
            "response": {criterion: {"satisfied": satisfied, "reason": "Because."}},
        }],
    }


def test_study_samples_and_entropies_use_all_six_draws():
    config = load_study(ROOT / "config/study_models.yaml")
    opus, sol = config.models[:2]
    records = [record(model.id, index, index % 2 == 0)
               for model in (opus, sol) for index in range(6)]
    table = build_table(records, {"strata": [], "papers": {}}, {"123": {"blood_mentions": 0}}, config.aliases)
    assert len(table) == 12 and table["sample_id"].is_unique
    assert table.loc[table["sample_index"] == 1, "v_sol"].tolist() == ["false", "false"]
    rows = [{**row, "source": f"s{row['sample_index']:02d}"}
            for item in records for row in record_rows(item)]
    assert entropy_table(rows)["n"].tolist() == [6, 6]
    assert not group_answers(rows)
    assert [len(group) for group in group_answers(rows, require_original=False).values()] == [6, 6]
