from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathology_report_extraction.graphs.build_stage_keyword_hierarchy_graphs import (  # noqa: E402
    find_stage_keywords,
    normalize_embedding_key,
)
from tools.extract_conch_sentence_features import conch_tokenize, load_conch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CONCH embeddings for stage keywords and NCIt/DO concept labels."
    )
    parser.add_argument(
        "--sentence_export_dir",
        type=Path,
        default=REPO_ROOT / "pathology_report_extraction" / "Output" / "sentence_exports_masked",
    )
    parser.add_argument(
        "--concept_dir",
        type=Path,
        default=REPO_ROOT / "pathology_report_extraction" / "Output" / "concept_annotations_ablation" / "ncit_do",
    )
    parser.add_argument("--datasets", nargs="+", default=["BRCA", "KIRC", "LUSC"])
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"F:\Tasks\Pathology_Report_Hierarchy_Graphs\_label_embeddings"),
    )
    parser.add_argument("--clam_root", type=Path, default=Path(r"F:\Tasks\CLAM-master"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\Tasks\CLAM-master\ckpts\pytorch_model.bin"))
    parser.add_argument("--model_cfg", type=str, default="conch_ViT-B-16")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_keyword_labels(sentence_export_dir: Path, datasets: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for dataset in datasets:
        dataset_dir = sentence_export_dir / dataset
        for path in sorted(dataset_dir.glob("*.json")):
            if path.name == "run_summary.json":
                continue
            payload = load_json(path)
            for sentence in payload.get("sentences", []) or []:
                for mention in find_stage_keywords(str(sentence)):
                    keyword = str(mention["keyword"])
                    labels.setdefault(keyword, str(mention.get("matched_text") or keyword).replace("_", " "))
    return labels


def collect_concept_labels(concept_dir: Path, datasets: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for dataset in datasets:
        dataset_dir = concept_dir / dataset
        for path in sorted(dataset_dir.glob("*.json")):
            if path.name == "run_summary.json":
                continue
            payload = load_json(path)
            for concept in payload.get("concepts", []) or []:
                concept_id = str(concept.get("concept_id") or "").strip()
                concept_name = str(concept.get("concept_name") or concept_id).strip()
                if not concept_id or not concept_name:
                    continue
                labels.setdefault(concept_id, concept_name)
                labels.setdefault(concept_name, concept_name)
    return labels


@torch.inference_mode()
def encode_labels(labels: dict[str, str], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    model, tokenizer = load_conch(args)
    keys = sorted(labels)
    embeddings: dict[str, torch.Tensor] = {}
    for start in range(0, len(keys), args.batch_size):
        batch_keys = keys[start : start + args.batch_size]
        texts = [labels[key] for key in batch_keys]
        token_ids = conch_tokenize(tokenizer, texts).to(args.device)
        features = model.encode_text(token_ids, normalize=True).detach().cpu().float()
        for key, text, feature in zip(batch_keys, texts, features):
            embeddings[key] = feature
            embeddings[normalize_embedding_key(key)] = feature
            embeddings[normalize_embedding_key(text)] = feature
        print(f"encoded {min(start + len(batch_keys), len(keys))}/{len(keys)}", flush=True)
    return embeddings


def save_map(path: Path, labels: dict[str, str], embeddings: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "embeddings": embeddings}, path)


def main() -> int:
    args = parse_args()
    keyword_labels = collect_keyword_labels(args.sentence_export_dir, [item.upper() for item in args.datasets])
    concept_labels = collect_concept_labels(args.concept_dir, [item.upper() for item in args.datasets])
    print(
        f"label embedding plan | keywords={len(keyword_labels)} | concepts={len(concept_labels)} | "
        f"output={args.output_dir}"
    )
    if args.dry_run:
        print("keyword sample:", list(keyword_labels.items())[:20])
        print("concept sample:", list(concept_labels.items())[:20])
        return 0

    keyword_embeddings = encode_labels(keyword_labels, args)
    save_map(args.output_dir / "stage_keyword_conch_embeddings.pt", keyword_labels, keyword_embeddings)
    concept_embeddings = encode_labels(concept_labels, args)
    save_map(args.output_dir / "ncit_do_concept_conch_embeddings.pt", concept_labels, concept_embeddings)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
