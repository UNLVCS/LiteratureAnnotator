#!/usr/bin/env python3
"""
RAG-based Data Labeling Script using LLM Providers

This script generates labeled data using RAG chains with multiple LLM providers
instead of just GPT. It processes papers from a queue and generates labeled
data based on the same criteria as the main.py webhook system.
"""

import os
import json
import sys
from typing import Any, Dict, List
from pathlib import Path

# Add the llm_providers to the path
sys.path.append(str(Path(__file__).parent.parent / "llm_providers"))

from llm_providers import (
    OpenAIProvider, 
    AnthropicProvider, 
    HuggingFaceProvider,
    Query
)

# Import the existing components
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from vector_db import VectorDb
from queue_helpers import (
    claim_next_paper,
    ack_paper,
    requeue_inflight,
    paper_queue_len
)


class RAGLabelingGenerator:
    """
    RAG-based labeling generator that uses multiple LLM providers
    """
    
    def __init__(self, provider_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize the RAG labeling generator
        
        Args:
            provider_configs: Dictionary mapping provider names to their configs
        """
        self.providers = {}
        self.setup_providers(provider_configs)
        
        # Setup vector store and embeddings
        self.vdb = VectorDb()
        self.embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vector_store = PineconeVectorStore(
            index=self.vdb.__get_index__(), 
            embedding=self.embedder, 
            namespace="article_upload_test_2"
        )
        
        # Load the RAG prompt
        self.prompt = hub.pull("rlm/rag-prompt")
        
        # Define the same criteria prompts as in main.py
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

            # 3) Sample size >= 50 (leniency note)
            """Criterion 3 – Sample Size
            If human study: determine if sample size n >= 50.
            - If stated n >= 50 → satisfied=true.
            - If < 50 → satisfied=false (note: can be relaxed later if other criteria are very strong).
            Return JSON only:
            {"criterion_3": {"satisfied": true/false, "reason": "<brief reason; include n if found>"}}""",

            # 4) Proteins as biomarkers (exclude gene/RNA/transcript/fragment focus)
            """Criterion 4 – Protein Biomarkers
            Decide if the study's biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
            - Satisfied if protein focus is central and recurrent.
            - Not satisfied if focus is genes/RNA/transcripts/fragments.
            Return JSON only:
            {"criterion_4": {"satisfied": true/false, "reason": "<brief reason>"}}""",

            # 5) Animal models exclusion (use human; flag patient cell-cultures)
            """Criterion 5 – Animal Models Exclusion
            Determine if the study uses animal models.
            - If animal models are used → satisfied=false.
            - If human data only → satisfied=true.
            - If using patient-derived cell cultures (not animals), note that explicitly.
            Return JSON only:
            {"criterion_5": {"satisfied": true/false, "reason": "<brief reason; note 'patient cell cultures' if applicable>"}}""",

            # 6) Blood as AD biomarker (not blood pressure)
            """Criterion 6 – Blood as AD Biomarker
            If 'blood' appears, decide if it is used as an AD biomarker (e.g., serum/plasma for amyloid/tau).
            - Exclude circulatory measures (e.g., blood pressure, hypertension, vascular health).
            Return JSON only:
            {"criterion_6": {"satisfied": true/false, "reason": "<brief reason>"}}""",

            # Final aggregation (single-shot option)
            """Final Classification – Aggregate
            Evaluate Criteria 1–6 and produce the final binary classification:
            - "relevant" if criteria strongly indicate AD-relevant original research (consider leniency on n<50 if others are strong).
            - "irrelevant" otherwise.
            Return JSON only and exactly this shape:
            {
            "criteria": {
                "criterion_1": {"satisfied": true/false, "reason": "<...>"},
                "criterion_2": {"satisfied": true/false, "reason": "<...>"},
                "criterion_3": {"satisfied": true/false, "reason": "<...>"},
                "criterion_4": {"satisfied": true/false, "reason": "<...>"},
                "criterion_5": {"satisfied": true/false, "reason": "<...>"},
                "criterion_6": {"satisfied": true/false, "reason": "<...>"}
            },
            "final_classification": "relevant/irrelevant",
            "justification": "<overall reasoning>"
            }"""
        ]
    
    def setup_providers(self, provider_configs: Dict[str, Dict[str, Any]]):
        """Setup LLM providers based on configuration"""
        for provider_name, config in provider_configs.items():
            try:
                if provider_name.lower() == "openai":
                    self.providers[provider_name] = OpenAIProvider(**config)
                elif provider_name.lower() == "anthropic":
                    self.providers[provider_name] = AnthropicProvider(**config)
                elif provider_name.lower() == "huggingface":
                    self.providers[provider_name] = HuggingFaceProvider(**config)
                else:
                    print(f"Warning: Unknown provider {provider_name}")
            except Exception as e:
                print(f"Failed to setup provider {provider_name}: {e}")
    
    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific paper from the vector store
        
        Args:
            paper_id: ID of the paper to retrieve chunks for
            
        Returns:
            List of document chunks with metadata
        """
        # Create a retriever with metadata filter for this specific paper
        filtered_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "filter": {"doc": paper_id},
                "k": 20  # Get more chunks to ensure we have the full paper
            }
        )
        
        # Retrieve documents
        docs = filtered_retriever.get_relevant_documents("")
        return docs
    
    def get_relevant_chunks_for_criteria(self, paper_id: str, criteria_query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a specific criteria query using vector similarity search
        
        Args:
            paper_id: ID of the paper to search within
            criteria_query: The specific criteria query to search for
            k: Number of most relevant chunks to retrieve
            
        Returns:
            List of most relevant document chunks for this criteria
        """
        # Create a retriever with metadata filter for this specific paper
        filtered_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "filter": {"doc": paper_id},
                "k": k  # Get top k most relevant chunks for this specific query
            }
        )
        
        # Use the criteria query to find most relevant chunks
        docs = filtered_retriever.get_relevant_documents(criteria_query)
        return docs
    
    def create_rag_query(self, paper_chunks: List[Dict[str, Any]], criteria_prompt: str) -> str:
        """
        Create a RAG query by combining paper chunks with the criteria prompt
        
        Args:
            paper_chunks: List of document chunks from the paper
            criteria_prompt: The criteria prompt to evaluate
            
        Returns:
            Combined query string for the LLM
        """
        # Combine all chunks into a single context
        context_parts = []
        for i, chunk in enumerate(paper_chunks):
            context_parts.append(f"=== Chunk {i+1} ===\n{chunk.page_content}\n")
        
        full_context = "\n".join(context_parts)
        
        # Create the RAG query with context instruction like in main.py
        rag_query = f"""
            Context from the research paper:
            {full_context}

            Task: {criteria_prompt}

            Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections.
            """
        return rag_query
    
    def process_paper_with_provider(self, paper_id: str, provider_name: str) -> Dict[str, Any]:
        """
        Process a single paper with a specific provider
        
        Args:
            paper_id: ID of the paper to process
            provider_name: Name of the provider to use
            
        Returns:
            Dictionary containing the results for this paper
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        provider = self.providers[provider_name]
        results = {
            "paper_id": paper_id,
            "provider": provider_name,
            "criteria_results": [],
            "final_classification": None,
            "chunks_processed": 0,
            "errors": []
        }
        
        try:
            # Get paper chunks
            paper_chunks = self.get_paper_chunks(paper_id)
            results["chunks_processed"] = len(paper_chunks)
            
            if not paper_chunks:
                results["errors"].append("No chunks found for paper")
                return results
            
            # Process each criteria
            for i, criteria_prompt in enumerate(self.criteria_prompts[:-1]):  # Exclude final aggregation
                try:
                    # Get relevant chunks for this specific criteria using vector similarity
                    relevant_chunks = self.get_relevant_chunks_for_criteria(paper_id, criteria_prompt, k=5)
                    
                    if not relevant_chunks:
                        results["errors"].append(f"No relevant chunks found for criterion {i+1}")
                        continue
                    
                    rag_query = self.create_rag_query(relevant_chunks, criteria_prompt)
                    
                    query = Query(
                        prompt=rag_query,
                        system_message="You are an expert research analyst. Analyze the provided paper content and respond with valid JSON only.",
                        temperature=0.1,  # Low temperature for consistent results
                        max_tokens=500
                    )
                    
                    response = provider.call_api(query)
                    
                    # Try to parse JSON response
                    try:
                        criteria_result = json.loads(response.content)
                        results["criteria_results"].append({
                            "criterion": f"criterion_{i+1}",
                            "prompt": criteria_prompt,
                            "response": criteria_result,
                            "raw_response": response.content,
                            "chunks_used": len(relevant_chunks)
                        })
                    except json.JSONDecodeError:
                        results["criteria_results"].append({
                            "criterion": f"criterion_{i+1}",
                            "prompt": criteria_prompt,
                            "response": None,
                            "raw_response": response.content,
                            "error": "Failed to parse JSON",
                            "chunks_used": len(relevant_chunks)
                        })
                        
                except Exception as e:
                    results["errors"].append(f"Criterion {i+1} failed: {str(e)}")
                    results["criteria_results"].append({
                        "criterion": f"criterion_{i+1}",
                        "prompt": criteria_prompt,
                        "response": None,
                        "error": str(e),
                        "chunks_used": 0
                    })
            
            # Process final aggregation
            try:
                final_prompt = self.criteria_prompts[-1]
                # For final aggregation, get more chunks since we need to consider the whole paper
                final_chunks = self.get_relevant_chunks_for_criteria(paper_id, final_prompt, k=10)
                
                if not final_chunks:
                    results["errors"].append("No relevant chunks found for final classification")
                    final_chunks = paper_chunks  # Fallback to all chunks
                
                rag_query = self.create_rag_query(final_chunks, final_prompt)
                
                query = Query(
                    prompt=rag_query,
                    system_message="You are an expert research analyst. Analyze the provided paper content and respond with valid JSON only.",
                    temperature=0.1,
                    max_tokens=1000
                )
                
                response = provider.call_api(query)
                
                try:
                    final_result = json.loads(response.content)
                    results["final_classification"] = {
                        **final_result,
                        "chunks_used": len(final_chunks)
                    }
                except json.JSONDecodeError:
                    results["final_classification"] = {
                        "error": "Failed to parse final classification JSON",
                        "raw_response": response.content,
                        "chunks_used": len(final_chunks)
                    }
                    
            except Exception as e:
                results["errors"].append(f"Final classification failed: {str(e)}")
                results["final_classification"] = {"error": str(e), "chunks_used": 0}
                
        except Exception as e:
            results["errors"].append(f"Paper processing failed: {str(e)}")
        
        return results
    
    def process_papers_batch(self, num_papers: int = 10, providers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of papers with specified providers
        
        Args:
            num_papers: Number of papers to process
            providers: List of provider names to use (defaults to all available)
            
        Returns:
            List of results for all processed papers
        """
        if providers is None:
            providers = list(self.providers.keys())
        
        all_results = []
        papers_processed = 0
        
        print(f"Starting batch processing of {num_papers} papers with providers: {providers}")
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
                for provider_name in providers:
                    print(f"  Processing with {provider_name}...")
                    result = self.process_paper_with_provider(paper_id, provider_name)
                    all_results.append(result)
                    
                    # Save intermediate results
                    self.save_results(all_results, f"intermediate_results_{papers_processed + 1}.json")
                
                # Acknowledge successful processing
                ack_paper(paper_id)
                papers_processed += 1
                
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
            filename = f"rag_labeling_results_{timestamp}.json"
        
        output_path = Path(__file__).parent / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """
    Main function to run the RAG labeling script
    """
    # Configuration for different providers
    provider_configs = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o",
            "temperature": 0.1
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.1
        },
        "huggingface": {
            "api_key": os.getenv("HUGGINGFACE_API_TOKEN"),
            "model": "microsoft/DialoGPT-medium",
            "temperature": 0.1
        }
    }
    
    # Filter out providers without API keys
    available_providers = {}
    for name, config in provider_configs.items():
        if config.get("api_key"):
            available_providers[name] = config
        else:
            print(f"Skipping {name} - no API key found")
    
    if not available_providers:
        print("No providers available. Please set API keys:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- HUGGINGFACE_API_TOKEN")
        return
    
    # Initialize the generator
    generator = RAGLabelingGenerator(available_providers)
    
    # Process papers
    print("Starting RAG-based labeling generation...")
    results = generator.process_papers_batch(
        num_papers=5,  # Adjust as needed
        providers=list(available_providers.keys())
    )
    
    # Save final results
    generator.save_results(results, "final_rag_labeling_results.json")
    
    print(f"\nCompleted processing {len(results)} paper-provider combinations")
    print("Results saved to final_rag_labeling_results.json")


if __name__ == "__main__":
    main()
