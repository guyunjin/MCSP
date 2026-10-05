import json
from pathlib import Path
from typing import Dict, Union


_DATASET_ALIASES = {
    "mintrec": "MIntRec",
    "mintrec2": "MIntRec2.0",
    "mintrec2.0": "MIntRec2.0",
    "meld": "MELD-DA",
    "meld-da": "MELD-DA",
}


def _canonical_dataset_name(dataset: str) -> str:
    name = str(dataset).strip()
    return _DATASET_ALIASES.get(name.lower(), name)


def load_intent_concepts(
    concepts_path: Union[str, Path],
    dataset: str,
    seed: int,
    num_labels: int,
) -> Dict[int, str]:
    'Load and validate intent concepts for a dataset seed.'
    path = Path(concepts_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Intent concepts file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        all_concepts = json.load(file)

    dataset_name = _canonical_dataset_name(dataset)
    if dataset_name not in all_concepts:
        available = ", ".join(sorted(all_concepts))
        raise KeyError(
            f"No intent concepts for dataset {dataset!r} "
            f"(resolved as {dataset_name!r}). Available datasets: {available}"
        )

    seed_key = str(int(seed))
    concepts_by_seed = all_concepts[dataset_name]
    if seed_key not in concepts_by_seed:
        available = ", ".join(sorted(concepts_by_seed, key=int))
        raise KeyError(
            f"No intent concepts for dataset {dataset_name!r}, seed={seed}. "
            f"Available seeds: {available}"
        )

    concepts = concepts_by_seed[seed_key]
    if not isinstance(concepts, list):
        raise TypeError(
            f"Concepts for dataset {dataset_name!r}, seed={seed} must be a list."
        )

    if len(concepts) != int(num_labels):
        raise ValueError(
            f"Expected {num_labels} concepts for dataset {dataset_name!r}, "
            f"seed={seed}, but found {len(concepts)}."
        )

    invalid_ids = [
        concept_id
        for concept_id, text in enumerate(concepts)
        if not isinstance(text, str) or not text.strip()
    ]
    if invalid_ids:
        raise ValueError(
            f"Empty or non-string concepts at IDs {invalid_ids} for "
            f"dataset {dataset_name!r}, seed={seed}."
        )

    return {concept_id: text.strip() for concept_id, text in enumerate(concepts)}
