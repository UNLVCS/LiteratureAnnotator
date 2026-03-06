import os
import json
import sys
from typing import Any, Dict, List
from pathlib import Path

from data_generation.response_standardizer import standardize_llm_response

from llm_providers.base import BaseLLMProvider, Query, LLMResponse
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.huggingface_provider import HuggingFaceProvider
from llm_providers.ollama_provider import OllamaProvider
from llm_providers.vllm_provider import VLLMProvider
from llm_providers.vllm_native_provider import VLLMNativeProvider
from dotenv import load_dotenv, find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from utilities.vector_db import VectorDb
from utilities.queue_helpers import (
    claim_next_paper,
    ack_paper,
    paper_queue_len,
    push_completed_paper,
    completed_papers_count,
    export_completed_papers_to_file
)
from minio import Minio
from io import BytesIO


load_dotenv(find_dotenv(), override=True)

client = Minio(
    os.getenv("MINIO_URL"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)
bucket_name = os.getenv("MINIO_BUCKET_NAME")
if not client.bucket_exists(bucket_name):
    print(f"Bucket {bucket_name} does not exist. Creating it...")
    client.make_bucket(bucket_name)
else:
    print(f"Bucket {bucket_name} already exists.")


class SampleGenerator:
    """
    Generates N stochastic samples per (paper, criterion) pair using a fixed
    prompt + context. Samples are used downstream for semantic entropy analysis.
    """

    def __init__(self, provider_configs: Dict[str, Dict[str, Any]], n_samples: int = 10, temperature: float = 1.0):
        """
        Args:
            provider_configs: Dictionary mapping provider names to their configs.
            n_samples: Number of times to call the LLM per (paper, criterion) pair.
            temperature: Sampling temperature; should be high (~1.0) to induce variation.
        """
        self.n_samples = n_samples
        self.temperature = temperature
        self.providers = {}
        self._setup_providers(provider_configs)

        self.vdb = VectorDb()
        self.embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vector_store = PineconeVectorStore(
            index=self.vdb.__get_index__(),
            embedding=self.embedder,
            namespace="V3_raw_pubmed_articles"
        )

        self.criteria_prompts = [
            # 1) Original research
            """Criterion 1 – Original Research
            Decide if the paper is an original research article (not a review, perspective, poster, or preprint).
            - Positive signals: data collection + statistical analysis (often in Methods).
            - Negative signals: clear mentions of review, perspective, poster, preprint.
            Return JSON only:
            {"criterion_1": {"satisfied": true/false, "reason": "<brief reason>"}}""",

            # 2) AD focus
            """Criterion 2 – AD Focus
            Decide if Alzheimer's Disease (AD) is the main focus (diagnosis, treatment, biomarkers, pathology; AD patients incl. MCI/at risk).
            - Include AD biomarkers: amyloid-beta, tau.
            - Exclude if focus is general neurodegeneration markers without AD specificity.
            Return JSON only:
            {"criterion_2": {"satisfied": true/false, "reason": "<brief reason>"}}""",

            # 3) Sample size >= 50
            """Criterion 3 – Sample Size
            If human study: determine if sample size n >= 50.
            - If stated n >= 50 → satisfied=true.
            - If < 50 → satisfied=false (note: can be relaxed later if other criteria are very strong).
            Return JSON only:
            {"criterion_3": {"satisfied": true/false, "reason": "<brief reason; include n if found>"}}""",

            # 4) Protein biomarkers
            """Criterion 4 – Protein Biomarkers
            Decide if the study's biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
            - Satisfied if protein focus is central and recurrent.
            - Not satisfied if focus is genes/RNA/transcripts/fragments.
            Return JSON only:
            {"criterion_4": {"satisfied": true/false, "reason": "<brief reason>"}}""",

            # 5) Animal models exclusion
            """Criterion 5 – Animal Models Exclusion
            Determine if the study uses animal models.
            - If animal models are used → satisfied=false.
            - If human data only → satisfied=true.
            - If using patient-derived cell cultures (not animals), note that explicitly.
            Return JSON only:
            {"criterion_5": {"satisfied": true/false, "reason": "<brief reason; note 'patient cell cultures' if applicable>"}}""",

            # 6) Blood as AD biomarker
            """Criterion 6 – Blood as AD Biomarker
            If 'blood' appears, decide if it is used as an AD biomarker (e.g., serum/plasma for amyloid/tau).
            - Exclude circulatory measures (e.g., blood pressure, hypertension, vascular health).
            Return JSON only:
            {"criterion_6": {"satisfied": true/false, "reason": "<brief reason>"}}""",
        ]

    def _setup_providers(self, provider_configs: Dict[str, Dict[str, Any]]):
        for provider, models in provider_configs.items():
            for model in models:
                if model['skip']:
                    continue
                if provider != "ollama" and not model.get("api_key"):
                    print(f"Skipping {model['model']} - no API key found")

                if provider == "openai":
                    self.providers[model['model']] = OpenAIProvider(**model)
                elif provider == "anthropic":
                    self.providers[model['model']] = AnthropicProvider(**model)
                elif provider == "huggingface":
                    self.providers[model['model']] = HuggingFaceProvider(**model)
                elif provider == "vllm":
                    self.providers[model['model']] = VLLMProvider(**model)
                elif provider == "vllm_native":
                    self.providers[model['model']] = VLLMNativeProvider(**model)
                elif provider == "ollama":
                    try:
                        temp = OllamaProvider(**model)
                        if temp.check_server_status():
                            self.providers[model['model']] = OllamaProvider(**model)
                            print(f"OLLAMA server is running - {model['model']} provider available")
                        else:
                            print(f"Skipping {model['model']} - OLLAMA server not running")
                    except Exception as e:
                        print(f"Skipping {model['model']} - OLLAMA setup failed: {e}")

    def _get_relevant_chunks(self, paper_id: str, criteria_query: str, k: int = 5) -> List[Any]:
        filtered_retriever = self.vector_store.as_retriever(
            search_kwargs={"filter": {"doc": paper_id}, "k": k}
        )
        return filtered_retriever.invoke(criteria_query)

    def _build_inference_query(self, chunks: List[Any], criteria_prompt: str) -> str:
        context = "\n".join(
            f"=== Chunk {i+1} ===\n{chunk.page_content}\n"
            for i, chunk in enumerate(chunks)
        )
        return (
            f"Context from the research paper:\n{context}\n\n"
            f"Task: {criteria_prompt}\n\n"
            "Consider ALL provided chunks of the paper when answering. "
            "Synthesize information from all relevant sections."
        )

    def generate_samples_for_paper(self, paper_id: str, provider_name: str) -> Dict[str, Any]:
        """
        For each criterion, retrieve context once then call the LLM self.n_samples
        times with the same prompt + context.

        Returns a dict structured as:
        {
            "paper_id": str,
            "provider": str,
            "n_samples": int,
            "temperature": float,
            "criteria": [
                {
                    "criterion": "criterion_1",
                    "prompt": str,
                    "context": str,          # fixed context used for all samples
                    "chunks_used": int,
                    "samples": [             # length == n_samples
                        {
                            "sample_index": int,
                            "raw_response": str,
                            "parsed_response": dict | None,
                            "parse_success": bool
                        },
                        ...
                    ]
                },
                ...
            ],
            "errors": []
        }
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")

        provider = self.providers[provider_name]
        result = {
            "paper_id": paper_id,
            "provider": provider_name,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "criteria": [],
            "errors": []
        }

        for i, criteria_prompt in enumerate(self.criteria_prompts):
            criterion_key = f"criterion_{i+1}"
            criterion_entry = {
                "criterion": criterion_key,
                "prompt": criteria_prompt,
                "context": "",
                "chunks_used": 0,
                "samples": []
            }

            try:
                chunks = self._get_relevant_chunks(paper_id, criteria_prompt, k=5)

                if not chunks:
                    result["errors"].append(f"No relevant chunks found for {criterion_key}")
                    result["criteria"].append(criterion_entry)
                    continue

                criterion_entry["chunks_used"] = len(chunks)
                inference_query = self._build_inference_query(chunks, criteria_prompt)
                criterion_entry["context"] = inference_query

                query = Query(
                    prompt=inference_query,
                    system_message=(
                        "You are an expert research analyst. "
                        "Analyze the provided paper content and respond with valid JSON only."
                    ),
                    temperature=self.temperature,
                    max_tokens=500
                )

                # Call the LLM n_samples times with the identical prompt + context
                for sample_idx in range(self.n_samples):
                    try:
                        response = provider.call_api(query)
                        parsed_json, cleaned_content, success = standardize_llm_response(response.content)
                        criterion_entry["samples"].append({
                            "sample_index": sample_idx,
                            "raw_response": response.content,
                            "cleaned_response": cleaned_content,
                            "parsed_response": parsed_json,
                            "parse_success": success
                        })
                    except Exception as e:
                        criterion_entry["samples"].append({
                            "sample_index": sample_idx,
                            "raw_response": None,
                            "cleaned_response": None,
                            "parsed_response": None,
                            "parse_success": False,
                            "error": str(e)
                        })
                        result["errors"].append(f"{criterion_key} sample {sample_idx} failed: {e}")

            except Exception as e:
                result["errors"].append(f"{criterion_key} failed: {e}")

            result["criteria"].append(criterion_entry)

        return result

    def generate_samples_batch(self, num_papers: int = 10, providers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Pull num_papers papers from the queue and generate N samples per
        (paper, criterion) for each specified provider.
        """
        if providers is None:
            providers = list(self.providers.keys())

        all_results = []
        papers_processed = 0

        print(f"Generating {self.n_samples} samples per criterion for {num_papers} papers")
        print(f"Temperature: {self.temperature} | Providers: {providers}")
        print(f"Queue length: {paper_queue_len()}")

        while papers_processed < num_papers:
            paper_id = claim_next_paper()
            if not paper_id:
                print("No more papers in queue")
                break

            print(f"\nPaper {paper_id} ({papers_processed + 1}/{num_papers})")

            try:
                for provider_name in providers:
                    print(f"  Provider: {provider_name} — generating {self.n_samples} samples per criterion...")
                    result = self.generate_samples_for_paper(paper_id, provider_name)
                    all_results.append(result)
                    self._save_to_minio(result)

                ack_paper(paper_id)
                push_completed_paper(paper_id)
                papers_processed += 1

            except Exception as e:
                print(f"Error on paper {paper_id}: {e}")
                continue

        return all_results

    def _save_to_minio(self, result: Dict[str, Any]):
        try:
            json_data = json.dumps(result).encode("utf-8")
            object_name = f"semantic_entropy_samples/{result['provider']}/{result['paper_id']}.json"
            client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=BytesIO(json_data),
                length=len(json_data),
                content_type="application/json"
            )
            print(f"  Saved → {bucket_name}/{object_name}")
        except Exception as e:
            print(f"  MinIO save failed for {result['paper_id']}: {e}")

    def save_results_local(self, results: List[Dict[str, Any]], filename: str = None):
        if filename is None:
            from datetime import datetime
            filename = f"semantic_entropy_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = Path(__file__).parent / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved locally → {output_path}")


def main():
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "llm_params" / "llm_params3.json") as f:
        provider_configs = json.load(f)

    generator = SampleGenerator(
        provider_configs=provider_configs,
        n_samples=10,
        temperature=1.0
    )

    results = generator.generate_samples_batch(
        num_papers=10,
        providers=["openai/gpt-oss:120b"]
    )

    generator.save_results_local(results, "semantic_entropy_samples.json")


if __name__ == "__main__":
    main()
