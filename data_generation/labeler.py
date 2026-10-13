#!/usr/bin/env python3
"""
Whole-paper Data Labeling Script using LLM Providers (sequential)

Single-process counterpart of labeler_mp.py for debugging: claims papers from the
Redis queue one at a time, labels each with every configured provider, and writes
results to a local JSON file instead of MinIO.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the parent directory to the path so we can import packages
sys.path.append(str(Path(__file__).parent.parent))

from labeling import SkipPaper, label_paper
from papers import PaperNotFoundError, load_paper_from_config
from utilities.queue_helpers import (
    claim_next_paper,
    ack_paper,
    requeue_inflight,
    paper_queue_len,
)


class LabelingGenerator:
    """
    Sequential whole-paper labeling generator that uses multiple LLM providers
    """

    def __init__(self):
        """Initialize the labeling generator using config from .env.yaml."""
        from config.app_config import load_app_config
        self.config = load_app_config()
        self.providers = self.config.get_providers_dict()

    def process_paper_with_provider(self, paper_id: str, provider_name: str) -> Dict[str, Any]:
        """
        Label a single paper with a specific provider over its full text.

        Raises:
            PaperNotFoundError / SkipPaper: the paper cannot be labeled.
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        paper = load_paper_from_config(paper_id, self.config)
        return label_paper(
            paper,
            provider_name,
            self.providers[provider_name],
            max_chars=self.config.papers.max_chars,
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
    """
    Main function to run the sequential labeling script
    """
    generator = LabelingGenerator()
    if not generator.providers:
        print("No providers available. Check .env.yaml llm_providers section and API keys.")
        return

    # Process papers
    print("Starting whole-paper labeling...")
    results = generator.process_papers_batch(
        num_papers=50,  # Adjust as needed
        providers=list(generator.providers.keys()),
    )

    # Save final results
    generator.save_results(results, "final_labeling_results.json")

    print(f"\nCompleted processing {len(results)} paper-provider combinations")
    print("Results saved to final_labeling_results.json")


if __name__ == "__main__":
    main()
