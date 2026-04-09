#!/usr/bin/env python3
"""
RAG-based Data Labeling Script using LLM Providers

This script generates labeled data using RAG chains with multiple LLM providers
instead of just GPT. It processes papers from a queue and generates labeled
data based on the same criteria as the main.py webhook system.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the parent directory to the path so we can import packages
sys.path.append(str(Path(__file__).parent.parent))

from response_standardizer import standardize_llm_response
from llm_providers.base import BaseLLMProvider, Query, LLMResponse
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from data_vectorize import VectorDb
from utilities.queue_helpers import (
    claim_next_paper,
    ack_paper,
    requeue_inflight,
    paper_queue_len,
)
from utilities.criteria import CRITERIA_PROMPTS


class RAGLabelingGenerator:
    """
    RAG-based labeling generator that uses multiple LLM providers
    """
    
    def __init__(self):
        """Initialize the RAG labeling generator using config from .env.yaml."""
        from config.app_config import load_app_config
        config = load_app_config()
        
        self.providers = config.get_providers_dict()
        
        # Setup vector store and embeddings
        self.vdb = VectorDb(pinecone_config=config.pinecone)
        self.embedder = OpenAIEmbeddings(
            model=config.embeddings.model,
            api_key=config.embeddings.api_key,
            dimensions=None if config.embeddings.model == "text-embedding-ada-002" else config.embeddings.dimensions,
        )
        self.vector_store = PineconeVectorStore(
            index=self.vdb.index,
            embedding=self.embedder,
            namespace=config.pinecone.namespace,
        )
        
        self.criteria_queries = CRITERIA_PROMPTS
    
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
        docs = filtered_retriever.invoke("")
        return docs
    
    def return_relevant_chunks(self, paper_id: str, criteria_query: str, k: int = 5) -> List[Dict[str, Any]]:
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
        # docs = filtered_retriever.get_relevant_documents(criteria_query)
        # pqa_chain = RetrievalQA.from_chain_type(
        #     llm = self.embedder,
        #     retriever = filtered_retriever,
        #     return_source_documents = True,
        #     chain_type = "stuff",
        #     chain_type_kwargs = {
        #         "prompt": self.prompt
        #     }
        # )

        # pqa_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # # Docs is a list of documents returned from the vector store, and likely contain sections that have answers for our criterias
        # docs = pqa_chain.invoke({
        #     "query": criteria_query,
        #     "context": "Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections."
        # })
        docs = filtered_retriever.invoke(criteria_query)
        return docs
    
    def create_inference_query(self, paper_chunks: List[Dict[str, Any]], criteria_prompt: str) -> str:
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
        inference_query = f"""
            Context from the research paper:
            {full_context}

            Task: {criteria_prompt}

            Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections.
            """
        return inference_query
    
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
        # Phase 1: retrieve chunks for every criterion (sequential vector searches).
        criterion_data = []
        for i, criteria_prompt in enumerate(self.criteria_queries):
            try:
                relevant_chunks = self.return_relevant_chunks(paper_id, criteria_prompt, k=5)
                full_context = "\n".join(
                    f"=== Chunk {j+1} ===\n{chunk.page_content}\n"
                    for j, chunk in enumerate(relevant_chunks)
                )
                criterion_data.append({
                    "idx": i,
                    "criteria_prompt": criteria_prompt,
                    "relevant_chunks": relevant_chunks,
                    "full_context": full_context,
                    "error": None,
                })
            except Exception as e:
                results["errors"].append(f"Criterion {i+1} chunk retrieval failed: {str(e)}")
                criterion_data.append({
                    "idx": i,
                    "criteria_prompt": criteria_prompt,
                    "relevant_chunks": [],
                    "full_context": "",
                    "error": str(e),
                })

        # Phase 2: batch all LLM calls for criteria that have chunks.
        valid = [cd for cd in criterion_data if not cd["error"] and cd["relevant_chunks"]]
        invalid = [cd for cd in criterion_data if cd["error"] or not cd["relevant_chunks"]]
        for cd in invalid:
            if not cd["error"]:
                results["errors"].append(f"No relevant chunks found for criterion {cd['idx']+1}")

        criterion_results: Dict[int, Dict] = {}
        if valid:
            queries = [
                Query(
                    prompt=self.create_inference_query(cd["relevant_chunks"], cd["criteria_prompt"]),
                    system_message="You are an expert research analyst. Analyze the provided paper content and respond with valid JSON only.",
                    temperature=0.1,
                    max_tokens=500,
                )
                for cd in valid
            ]
            try:
                responses = provider.call_api_batch(queries)
                for cd, response in zip(valid, responses):
                    parsed_json, cleaned_content, success = standardize_llm_response(response.content)
                    criterion_results[cd["idx"]] = {
                        "criterion": f"criterion_{cd['idx']+1}",
                        "prompt": cd["criteria_prompt"],
                        "response": parsed_json if (success and parsed_json) else None,
                        "raw_response": response.content,
                        "cleaned_response": cleaned_content,
                        "error": None if (success and parsed_json) else "Failed to parse JSON",
                        "chunks_used": len(cd["relevant_chunks"]),
                    }
            except Exception as e:
                for cd in valid:
                    results["errors"].append(f"Criterion {cd['idx']+1} LLM call failed: {str(e)}")
                    criterion_results[cd["idx"]] = {
                        "criterion": f"criterion_{cd['idx']+1}",
                        "prompt": cd["criteria_prompt"],
                        "response": None,
                        "error": str(e),
                        "chunks_used": 0,
                    }

        for cd in invalid:
            criterion_results[cd["idx"]] = {
                "criterion": f"criterion_{cd['idx']+1}",
                "prompt": cd["criteria_prompt"],
                "response": None,
                "error": cd["error"] or "No relevant chunks found",
                "chunks_used": 0,
            }

        # Phase 3: collect in strict criterion order.
        results["criteria_results"] = [
            criterion_results[i] for i in range(len(self.criteria_queries)) if i in criterion_results
        ]
            
            # Process final aggregation
            # try:
                # final_prompt = self.criteria_queries[-1]
                # For final aggregation, get more chunks since we need to consider the whole paper
                # final_chunks = self.get_relevant_chunks_for_criteria(paper_id, final_prompt, k=10)
                # 
                # if not final_chunks:
                    # results["errors"].append("No relevant chunks found for final classification")
                    # final_chunks = paper_chunks  # Fallback to all chunks
                # 
                # rag_query = self.create_rag_query(final_chunks, final_prompt)
                # 
                # query = Query(
                    # prompt=rag_query,
                    # system_message="You are an expert research analyst. Analyze the provided paper content and respond with valid JSON only.",
                    # temperature=0.1,
                    # max_tokens=1000
                # )
                # 
                # response = provider.call_api(query)
                # 
                # try:
                    # final_result = json.loads(response.content)
                    # results["final_classification"] = {
                        # **final_result,
                        # "chunks_used": len(final_chunks)
                    # }
                # except json.JSONDecodeError:
                    # results["final_classification"] = {
                        # "error": "Failed to parse final classification JSON",
                        # "raw_response": response.content,
                        # "chunks_used": len(final_chunks)
                    # }
                    
            # except Exception as e:
                # results["errors"].append(f"Final classification failed: {str(e)}")
                # results["final_classification"] = {"error": str(e), "chunks_used": 0}
                
        # except Exception as e:
        #     results["errors"].append(f"Paper processing failed: {str(e)}")
        
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
    generator = RAGLabelingGenerator()
    if not generator.providers:
        print("No providers available. Check .env.yaml llm_providers section and API keys.")
        return
    
    # Process papers
    print("Starting RAG-based labeling generation...")
    results = generator.process_papers_batch(
        num_papers=50,  # Adjust as needed
        providers=list(generator.providers.keys()),
    )
    
    # Save final results
    generator.save_results(results, "final_rag_labeling_results.json")
    
    print(f"\nCompleted processing {len(results)} paper-provider combinations")
    print("Results saved to final_rag_labeling_results.json")


if __name__ == "__main__":
    main()
