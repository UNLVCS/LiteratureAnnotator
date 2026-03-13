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
            Your task: Determine whether this paper reports original empirical research — meaning it
            collected and analyzed new primary data — rather than being a secondary publication type.

            Reasoning steps (work through these before deciding):
            1. Does the paper have a Methods section describing data collection, participant recruitment,
               sample processing, or experimental procedures? (positive signal)
            2. Does it report statistical analysis of primary data — e.g., regression, t-tests, ANOVA,
               survival analysis, correlation? (positive signal)
            3. Does the text contain terms like "systematic review," "meta-analysis," "literature review,"
               "perspective," "commentary," "editorial," "poster," "preprint," or "protocol"? (negative signal)

            INCLUDE (satisfied=true): Paper describes original data collection and analysis, even if it
            briefly reviews prior literature.
            EXCLUDE (satisfied=false): Paper is primarily a review, meta-analysis, perspective, editorial,
            commentary, protocol, poster, or preprint with no new primary data.

            When uncertain: If a Methods and Results section with primary data are present, default to
            satisfied=true.
            Return JSON only:
            {"criterion_1": {"satisfied": true/false, "reason": "<one sentence citing specific evidence>"}}""",

            # 2) AD focus
            """Criterion 2 – AD Focus
            Your task: Determine whether this paper is EITHER (a) primarily focused on Alzheimer's
            Disease (AD), OR (b) substantially involves amyloid-beta (Aβ) measurement, presence, or
            evaluation — even if AD is not the central topic. This is an OR condition, not AND.

            Reasoning steps (work through these before deciding):
            1. Is AD the primary subject of the study? Consider diagnosis, treatment, prevention,
               pathology, or biomarker measurement in AD, MCI, or at-risk populations.
            2. If AD is not the primary subject, does the paper still measure or evaluate amyloid-beta
               (Aβ40, Aβ42, amyloid plaques, amyloid PET) in a meaningful way — not a single passing mention?
            3. Does the paper study a non-AD population (e.g., Down Syndrome, cognitively normal adults)
               but include amyloid-beta as a key component? If yes → include.
            4. Is AD or amyloid mentioned only once (e.g., one sentence in the introduction) with no
               further study of it? If yes → exclude.
            5. Is the paper about general neurodegeneration or tau pathology only, without any AD or
               amyloid-beta focus? Note: tau alone (t-tau, p-tau) is NOT sufficient for inclusion —
               tau changes occur across many neurodegenerative diseases.

            INCLUDE (satisfied=true) if ANY of the following is true:
            - AD is the primary subject (diagnosis, treatment, pathology, biomarkers, at-risk populations).
            - Amyloid-beta is measured or evaluated as a meaningful component of the study.
            - A non-AD population is studied but amyloid-beta is a key endpoint or stratification variable.

            EXCLUDE (satisfied=false) if ALL of the following are true:
            - AD and amyloid-beta are absent or only incidentally mentioned (≤1 sentence).
            - The paper focuses on general neurodegeneration, tau-only pathology, or unrelated disease.

            Key logic: AD-focus OR amyloid-beta involvement → include. Neither → exclude.
            Return JSON only:
            {"criterion_2": {"satisfied": true/false, "reason": "<one sentence citing specific evidence>"}}""",

            # 3) Sample size >= 50
            """Criterion 3 – Sample Size
            Your task: Determine whether the study's human sample size meets the minimum threshold of n ≥ 50.

            Reasoning steps (work through these before deciding):
            1. Find the total number of human participants. Check the abstract, Methods (participants
               section), Results, and any tables.
            2. If participants are split into groups (e.g., AD patients + controls), sum all groups to
               get the total N.
            3. If the paper reports multiple independent cohorts, use the primary/discovery cohort's N.
            4. If the paper is not a human study (animal-only, in vitro only), mark satisfied=false and
               note "not a human study" in the reason.

            INCLUDE (satisfied=true): Total human N ≥ 50, either stated explicitly or calculable from
            reported group sizes.
            EXCLUDE (satisfied=false): Total N < 50, N cannot be determined from the text, or the study
            does not involve human participants.

            When uncertain: If N appears to be above 50 but is ambiguously reported, lean toward
            satisfied=true and note the uncertainty.
            Return JSON only:
            {"criterion_3": {"satisfied": true/false, "reason": "<one sentence; include N if found>"}}""",

            # 4) Targeted biomarker methodology
            """Criterion 4 – Targeted Biomarker Methodology
            Your task: Determine whether the study uses a targeted (hypothesis-driven) measurement
            approach — as opposed to a purely exploratory/discovery-wide approach. The biomarker domain
            can be proteins, transcripts, metabolites, genes, or imaging; what matters is whether the
            methodology is targeted or exploratory.

            Reasoning steps (work through these before deciding):
            1. Identify the primary measurement methods used (check Methods section).
            2. Do they match targeted methods (see INCLUDE list)? If yes → satisfied=true.
            3. Are they purely exploratory/discovery-wide (see EXCLUDE list) with no targeted component? If yes → satisfied=false.
            4. Does the study mix targeted and exploratory methods? If yes → satisfied=true (hybrid rule).

            INCLUDE (satisfied=true) if the study uses ANY of the following targeted methods:
            - Protein/peptide assays: ELISA, Simoa, MSD, ECL, immunoassay, targeted mass spectrometry,
              predefined proteomic panels.
            - Genetic/transcriptomic (targeted): qPCR, ddPCR for specific transcripts, targeted RNA
              panels (e.g., NanoString), APOE genotyping, specific variant assays, candidate-gene analyses.
            - Imaging: specific PET tracers (amyloid PET, tau PET, FDG-PET), predefined MRI/imaging
              endpoints.
            - Metabolomics: predefined panels targeting specific analytes.
            - Hybrid: any combination of targeted + exploratory methods.

            EXCLUDE (satisfied=false) ONLY if the study is purely exploratory with NO targeted component:
            - GWAS, whole-genome sequencing (WGS), whole-exome sequencing (WES) as the primary analysis.
            - Untargeted/shotgun proteomics (discovery DDA), untargeted metabolomics/lipidomics.
            - Whole transcriptome RNA-seq, scRNA-seq, single-cell multi-omics, unbiased transcriptomic screens.
            - Broad "multi-omics discovery," "agnostic screen," or "unbiased" studies with no predefined targets.
            - Exosome full sequencing or full small-RNA sequencing studies.

            Key logic: targeted OR hybrid → include; purely exploratory with zero targeted component → exclude.
            Return JSON only:
            {"criterion_4": {"satisfied": true/false, "reason": "<one sentence citing the specific method>"}}""",

            # 5) Human participants only
            """Criterion 5 – Human Participants Only
            Your task: Determine whether the study's primary subjects are human participants (not
            animal models or purely non-human in vitro systems).

            Reasoning steps (work through these before deciding):
            1. Check the Methods for animal species: "mouse," "mice," "rat," "murine," "transgenic
               model," "APP/PS1," "5xFAD," "3xTg," "zebrafish," "non-human primate," etc.
               If present as the primary subject → satisfied=false.
            2. Check whether the study involves human participants: recruited subjects, patient cohorts,
               clinical samples, or human volunteers.
            3. Are patient-derived cell cultures used (e.g., iPSC-derived neurons, human CSF-derived
               cells)? These are acceptable — they are NOT animal models → satisfied=true.
            4. Does the study use both humans and animals? Evaluate whether the human component is
               primary. If so, lean toward satisfied=true.

            INCLUDE (satisfied=true): Study's primary subjects are human participants, OR human-derived
            in vitro systems (patient cell cultures, human iPSCs).
            EXCLUDE (satisfied=false): Study's primary subjects are animal models (mice, rats, non-human
            primates, etc.), even if minimal human data is also reported.

            When uncertain: If human data is the primary analysis and animal data is only a secondary
            validation, default to satisfied=true.
            Return JSON only:
            {"criterion_5": {"satisfied": true/false, "reason": "<one sentence; note if patient cell cultures or hybrid design>"}}""",

            # 6) Blood-based AD biomarker
            """Criterion 6 – Blood-Based AD Biomarker
            Your task: Determine whether the study uses blood (serum, plasma, whole blood, or
            blood-derived fractions) as a source for measuring AD-relevant biomarkers.

            Reasoning steps (work through these before deciding):
            1. Does the Methods section mention blood draw, serum, or plasma as a sample type?
            2. What is the blood used to measure? Look for AD-relevant analytes: amyloid-beta (Aβ40,
               Aβ42), phosphorylated tau (p-tau181, p-tau217), neurofilament light (NfL), GFAP,
               or other AD biomarker proteins.
            3. Is blood used only for non-AD-relevant measures? Examples that do NOT satisfy:
               blood pressure, lipid panels for cardiovascular risk, CBC, metabolic panels,
               glucose/insulin levels (unless in an AD biomarker context).
            4. Is blood used for APOE genotyping or other AD-relevant genetic analyses?
               This satisfies the criterion.
            5. Is blood not used in this study at all (e.g., CSF-only, imaging-only, urine-only)?
               If so → satisfied=false.

            INCLUDE (satisfied=true): Blood/serum/plasma is collected and used to measure AD-relevant
            biomarkers (amyloid, tau, NfL, GFAP, APOE, other AD proteins or genetic markers).
            EXCLUDE (satisfied=false): Blood is used only for cardiovascular/metabolic measures
            unrelated to AD biomarkers, or blood is not collected/used in the study.

            When uncertain: If blood is collected and analyzed in any AD-relevant context, default to
            satisfied=true.
            Return JSON only:
            {"criterion_6": {"satisfied": true/false, "reason": "<one sentence citing the specific blood measure>"}}""",
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
                        "You are a senior biomedical research analyst with deep expertise in "
                        "Alzheimer's Disease (AD) and neurodegenerative disease research. "
                        "Your role is to screen scientific papers for inclusion in a curated "
                        "AD biomarker study dataset.\n\n"
                        "For each query you will receive excerpts (chunks) from a single research "
                        "paper and one specific inclusion criterion to evaluate.\n\n"
                        "Your responsibilities:\n"
                        "1. Read every chunk carefully and synthesize information across all "
                        "sections (abstract, methods, results, discussion).\n"
                        "2. Apply the criterion exactly as specified — do not add or remove conditions.\n"
                        "3. Reason through the evidence methodically before reaching a conclusion.\n"
                        "4. When evidence is ambiguous or information is missing, err toward "
                        "inclusion (satisfied=true) rather than exclusion, unless the criterion "
                        "is clearly not met.\n"
                        "5. Respond with valid JSON only — no preamble, markdown, or text outside "
                        "the JSON object.\n\n"
                        "Your 'reason' field must be a single concise sentence citing specific "
                        "evidence from the paper (e.g., method names, reported N, specific terms "
                        "found or not found)."
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
