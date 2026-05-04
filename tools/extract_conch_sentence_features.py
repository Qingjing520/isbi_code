from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode exported pathology report sentences with the local CONCH text encoder."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing sentence export JSON files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for case-level .pt tensors.")
    parser.add_argument("--clam_root", type=Path, default=Path(r"F:\Tasks\CLAM-master"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\Tasks\CLAM-master\ckpts\pytorch_model.bin"))
    parser.add_argument("--model_cfg", type=str, default="conch_ViT-B-16")
    parser.add_argument("--label_csv", type=Path, default=None, help="Optional label CSV used to restrict cases.")
    parser.add_argument("--only_missing_from_label", action="store_true", help="Only encode label cases missing in output_dir.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def case_id_from_text(text: str) -> str:
    match = CASE_ID_RE.search(str(text or ""))
    if match:
        return match.group(1).upper()
    return Path(str(text)).stem[:12].upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_label_cases(label_csv: Path | None) -> set[str]:
    if label_csv is None:
        return set()
    with label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("case_id") or "").strip().upper() for row in rows if str(row.get("case_id") or "").strip()}


def collect_exports(input_dir: Path) -> dict[str, Path]:
    exports: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"run_summary.json"} or path.name.startswith("pipeline_run_summary_"):
            continue
        payload = load_json(path)
        case_id = str(payload.get("document_id") or "").strip().upper() or case_id_from_text(path.name)
        exports.setdefault(case_id, path)
    return exports


def load_sentences(path: Path) -> list[str]:
    payload = load_json(path)
    sentences = payload.get("sentences") or []
    cleaned = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    if not cleaned:
        raise ValueError(f"No non-empty sentences in {path}")
    return cleaned


def load_conch(args: argparse.Namespace):
    sys.path.insert(0, str(args.clam_root))
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

    model = create_model_from_pretrained(
        args.model_cfg,
        checkpoint_path=str(args.checkpoint),
        device=args.device,
        return_transform=False,
    )
    model.eval()
    tokenizer = get_tokenizer()
    return model, tokenizer


def conch_tokenize(tokenizer: Any, texts: list[str]) -> torch.Tensor:
    kwargs = {
        "max_length": 127,
        "add_special_tokens": True,
        "return_token_type_ids": False,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    if hasattr(tokenizer, "batch_encode_plus"):
        tokens = tokenizer.batch_encode_plus(texts, **kwargs)
    else:
        tokens = tokenizer(texts, **kwargs)
    input_ids = tokens["input_ids"]
    return F.pad(input_ids, (0, 1), value=tokenizer.pad_token_id)


@torch.inference_mode()
def encode_sentences(
    sentences: list[str],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        token_ids = conch_tokenize(tokenizer, batch).to(device)
        features = model.encode_text(token_ids, normalize=True).detach().cpu().float()
        chunks.append(features)
    return torch.cat(chunks, dim=0)


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"CONCH checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = collect_exports(args.input_dir)
    label_cases = load_label_cases(args.label_csv)
    existing_cases = {path.stem.upper() for path in args.output_dir.glob("*.pt")}

    target_cases = set(exports)
    if label_cases:
        target_cases &= label_cases
    if args.only_missing_from_label:
        if not label_cases:
            raise ValueError("--only_missing_from_label requires --label_csv")
        target_cases = label_cases - existing_cases
        missing_exports = sorted(target_cases - set(exports))
        if missing_exports:
            print(f"WARNING: {len(missing_exports)} target cases have no sentence export JSON. Sample={missing_exports[:10]}")
        target_cases &= set(exports)
    elif not args.overwrite:
        target_cases -= existing_cases

    ordered_cases = sorted(target_cases)
    if args.limit > 0:
        ordered_cases = ordered_cases[: args.limit]

    print(
        "CONCH sentence feature plan | "
        f"exports={len(exports)} | existing={len(existing_cases)} | targets={len(ordered_cases)} | "
        f"device={args.device} | output={args.output_dir}"
    )
    if args.dry_run:
        for case_id in ordered_cases[:50]:
            print(f"DRY {case_id} <- {exports[case_id].name}")
        return 0

    model, tokenizer = load_conch(args)
    written = 0
    failed: list[tuple[str, str]] = []
    for idx, case_id in enumerate(ordered_cases, start=1):
        out_path = args.output_dir / f"{case_id}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        try:
            sentences = load_sentences(exports[case_id])
            features = encode_sentences(
                sentences,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                batch_size=args.batch_size,
            )
            torch.save(features, out_path)
            written += 1
            print(f"[{idx}/{len(ordered_cases)}] saved {out_path.name} shape={tuple(features.shape)}")
        except Exception as exc:
            failed.append((case_id, str(exc)))
            print(f"[{idx}/{len(ordered_cases)}] FAILED {case_id}: {exc}")

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "label_csv": str(args.label_csv) if args.label_csv else "",
        "targets": len(ordered_cases),
        "written": written,
        "failed": [{"case_id": case_id, "error": error} for case_id, error in failed],
    }
    (args.output_dir / "conch_sentence_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Done | written={written} | failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
