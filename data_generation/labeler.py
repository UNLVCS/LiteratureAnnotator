#!/usr/bin/env python3
"""
Whole-paper Data Labeling Script using LLM Providers (sequential)

Single-process counterpart of labeler_mp.py: labels papers without the Redis queue.
Results are written locally, and to MinIO with --upload.

    # Specific papers, no Redis queue: one JSON per paper and provider in --out
    python data_generation/labeler.py --pmids 41646005,40320869 --raw-prefix pilot_2026_10

    # Every paper in a MinIO folder, results uploaded to minio.synthetic_data_bucket,
    # skipping papers that already have a result for that model (restartable; SLURM)
    python data_generation/labeler.py --from-prefix pilot_2026_10 --limit 30 --upload --skip-existing

    # Claim papers from the Redis queue (same queue as labeler_mp.py)
    python data_generation/labeler.py --num-papers 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import packages
sys.path.append(str(Path(__file__).parent.parent))

from labeling import SkipPaper, label_paper, result_object_name
from papers import PaperNotFoundError, load_paper_from_config


class LabelingGenerator:
    """
    Sequential whole-paper labeling generator that uses multiple LLM providers
    """

    def __init__(self, raw_prefix: Optional[str] = None):
        """Initialize the labeling generator using config from .env.yaml."""
        from config.app_config import load_app_config
        self.config = load_app_config()
        if raw_prefix is not None:
            self.config.papers.raw_prefix = raw_prefix
        self.providers = self.config.get_providers_dict()
        from minio import Minio

        m = self.config.minio
        self.minio = Minio(m.url, access_key=m.access_key, secret_key=m.secret_key, secure=m.secure)

    def raw_prefix(self) -> str:
        prefix = self.config.papers.raw_prefix
        return prefix if prefix is not None else (self.config.bioc_download.object_prefix or "")

    def list_prefix_pmids(self) -> List[str]:
        """PMIDs of every raw article under the configured raw prefix."""
        prefix = self.raw_prefix().strip("/")
        objects = self.minio.list_objects(
            self.config.minio.raw_articles_bucket, prefix=f"{prefix}/" if prefix else "", recursive=False
        )
        return sorted(
            Path(o.object_name).stem for o in objects if o.object_name.endswith(".json") and not o.is_dir
        )

    def result_exists(self, provider_name: str, pmid: str) -> bool:
        from minio.error import S3Error

        name = result_object_name({"provider": provider_name, "paper_id": pmid}, self.config.minio.output_prefix)
        try:
            self.minio.stat_object(self.config.minio.synthetic_data_bucket, name)
            return True
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject"):
                return False
            raise

    def upload(self, result: Dict[str, Any]) -> str:
        from io import BytesIO

        name = result_object_name(result, self.config.minio.output_prefix)
        data = json.dumps(result, indent=2).encode("utf-8")
        self.minio.put_object(
            self.config.minio.synthetic_data_bucket, name, BytesIO(data), len(data), content_type="application/json"
        )
        return f"{self.config.minio.synthetic_data_bucket}/{name}"

    def process_paper_with_provider(self, paper_id: str, provider_name: str) -> Dict[str, Any]:
        """
        Label a single paper with a specific provider over its full text.

        Raises:
            PaperNotFoundError / SkipPaper: the paper cannot be labeled.
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        provider = self.providers[provider_name]
        paper = load_paper_from_config(paper_id, self.config, client=self.minio)
        # Per-model settings from .env.yaml win: reasoning models (e.g. gpt-oss) need far
        # more output tokens than the 1000 that suffice for the JSON answer itself.
        return label_paper(
            paper,
            provider_name,
            provider,
            max_chars=self.config.papers.max_chars,
            temperature=provider.config.get("temperature", 0.1),
            max_tokens=provider.config.get("max_tokens") or 1000,
        )

    def process_papers_batch(self, num_papers: int = 10, providers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of papers with specified providers

        Args:
            num_papers: Number of papers to process
            providers: List of provider names to use (defaults to all available)

        Returns:
            List of results for all processed papers
        """
        from utilities.queue_helpers import ack_paper, claim_next_paper, paper_queue_len, requeue_inflight

        configured_provider_names = list(self.providers.keys())
        if providers is None:
            provider_names = configured_provider_names
        else:
            provider_names = [name for name in providers if name in self.providers]
            missing_providers = [name for name in providers if name not in self.providers]
            for missing in missing_providers:
                print(f"Skipping unknown/unconfigured provider: {missing}")

        if not provider_names:
            print("No providers available for processing")
            return []

        all_results = []
        papers_processed = 0

        print(f"Starting batch processing of {num_papers} papers with providers: {provider_names}")
        print(f"Queue length: {paper_queue_len()}")

        while papers_processed < num_papers:
            # Claim next paper
            paper_id = claim_next_paper()
            if not paper_id:
                print("No more papers in queue")
                break

            print(f"\nProcessing paper {paper_id} ({papers_processed + 1}/{num_papers})")

            try:
                # Process with each provider
                for provider_name in provider_names:
                    print(f"  Processing with {provider_name}...")
                    result = self.process_paper_with_provider(paper_id, provider_name)
                    all_results.append(result)

                    # Save intermediate results
                    self.save_results(all_results, f"intermediate_results_{papers_processed + 1}.json")

                # Acknowledge successful processing
                ack_paper(paper_id)
                papers_processed += 1

            except (PaperNotFoundError, SkipPaper) as e:
                # Retrying cannot help: the raw article is missing or empty.
                print(f"Skipping paper {paper_id}: {e}")
                ack_paper(paper_id)
                continue

            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                requeue_inflight(paper_id)
                continue

        return all_results

    def label_pmids(
        self,
        pmids: List[str],
        provider_names: List[str],
        out_dir: Path,
        upload: bool = False,
        skip_existing: bool = False,
    ) -> List[Dict[str, Any]]:
        """Label the given papers with each provider, writing ``{out_dir}/{provider}/{pmid}.json``.

        With ``upload``, results are also written to MinIO, except those whose LLM call
        failed (``result["errors"]``), so a rerun with ``skip_existing`` retries them.
        """
        results = []
        for i, pmid in enumerate(pmids, 1):
            for provider_name in provider_names:
                if skip_existing and self.result_exists(provider_name, pmid):
                    print(f"[{i}/{len(pmids)}] {pmid} with {provider_name}: already labeled, skipping")
                    continue
                print(f"[{i}/{len(pmids)}] {pmid} with {provider_name}...", flush=True)
                try:
                    result = self.process_paper_with_provider(pmid, provider_name)
                except (PaperNotFoundError, SkipPaper) as e:
                    print(f"  skipped: {e}")
                    break
                path = out_dir / provider_name / f"{pmid}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2))
                usage = [c.get("usage") or {} for c in result["criteria_results"]]
                cached = sum(u.get("cache_read_input_tokens", u.get("cached_prompt_tokens", 0)) for u in usage)
                errors = [c["criterion"] for c in result["criteria_results"] if c.get("error")]
                print(f"  saved {path}  cached_input_tokens={cached}  criterion_errors={errors or 'none'}"
                      + (f"  CALL ERRORS: {result['errors']}" if result["errors"] else ""), flush=True)
                if upload and not result["errors"]:
                    print(f"  uploaded {self.upload(result)}", flush=True)
                results.append(result)
        return results

    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """
        Save results to a JSON file

        Args:
            results: List of results to save
            filename: Optional filename (defaults to timestamped filename)
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"labeling_results_{timestamp}.json"

        output_path = Path(__file__).parent / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sequential whole-paper labeling (debugging).")
    parser.add_argument("--pmids", help="Comma-separated PMIDs to label directly (no Redis queue).")
    parser.add_argument("--pmids-file", help="File with one PMID per line (no Redis queue).")
    parser.add_argument("--raw-prefix", help="MinIO folder holding the papers, e.g. pilot_2026_10.")
    parser.add_argument("--providers", help="Comma-separated model names to use (default: all configured).")
    parser.add_argument("--out", default=str(Path(__file__).parent / "dryrun"), help="Output dir for --pmids mode.")
    parser.add_argument("--from-prefix", help="Label every paper in this MinIO raw-articles folder (no Redis queue).")
    parser.add_argument("--limit", type=int, help="With --pmids-file/--from-prefix: only the first N papers.")
    parser.add_argument("--upload", action="store_true", help="Also write results to minio.synthetic_data_bucket.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip papers already labeled in MinIO by that model.")
    parser.add_argument("--num-papers", type=int, default=50, help="Queue mode: papers to claim.")
    args = parser.parse_args()

    generator = LabelingGenerator(raw_prefix=args.from_prefix or args.raw_prefix)
    if not generator.providers:
        print("No providers available. Check .env.yaml llm_providers section and API keys.")
        return
    provider_names = list(generator.providers)
    if args.providers:
        wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
        provider_names = [p for p in wanted if p in generator.providers]
        for p in set(wanted) - set(provider_names):
            print(f"Skipping unknown/unconfigured provider: {p}")

    pmids = [p.strip() for p in (args.pmids or "").split(",") if p.strip()]
    if args.pmids_file:
        pmids += [line.strip() for line in Path(args.pmids_file).read_text().splitlines() if line.strip()]
    if args.from_prefix:
        pmids += generator.list_prefix_pmids()
    if args.limit:
        pmids = pmids[: args.limit]
    if pmids:
        results = generator.label_pmids(
            pmids, provider_names, Path(args.out), upload=args.upload, skip_existing=args.skip_existing
        )
        print(f"\nLabeled {len(results)} paper-provider combinations into {args.out}")
        return

    print("Starting whole-paper labeling from the Redis queue...")
    results = generator.process_papers_batch(num_papers=args.num_papers, providers=provider_names)
    generator.save_results(results, "final_labeling_results.json")
    print(f"\nCompleted processing {len(results)} paper-provider combinations")


if __name__ == "__main__":
    main()
