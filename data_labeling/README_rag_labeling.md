# RAG-based Data Labeling Script

This script generates labeled data using RAG (Retrieval-Augmented Generation) chains with multiple LLM providers instead of just GPT. It processes papers from a queue and generates labeled data based on the same criteria as the main webhook system.

## Features

- **Multi-provider support**: Works with OpenAI, Anthropic, and Hugging Face models
- **RAG-based processing**: Uses vector store to retrieve relevant paper chunks
- **Batch processing**: Processes multiple papers efficiently
- **Queue integration**: Works with the existing Redis-based paper queue
- **Comprehensive criteria**: Evaluates papers on 6 different criteria plus final classification
- **Error handling**: Robust error handling with requeuing of failed papers
- **Intermediate saves**: Saves results incrementally to prevent data loss

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key" 
   export HUGGINGFACE_API_TOKEN="your-hf-token"
   ```

3. **Ensure Redis is running**:
   ```bash
   redis-server
   ```

4. **Make sure papers are in the queue**:
   ```bash
   python seed_queue.py  # or your queue seeding script
   ```

## Usage

### Basic Usage

```bash
python rag_labeling_script.py
```

### Configuration

The script automatically detects available providers based on API keys. You can modify the configuration in the `main()` function:

```python
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
    }
}
```

### Customizing Processing

You can customize the processing by modifying the `RAGLabelingGenerator` class:

```python
# Process more papers
results = generator.process_papers_batch(
    num_papers=20,  # Increase number of papers
    providers=["openai", "anthropic"]  # Use specific providers
)
```

## Output Format

The script generates JSON files with the following structure:

```json
{
  "paper_id": "paper_123",
  "provider": "openai",
  "criteria_results": [
    {
      "criterion": "criterion_1",
      "prompt": "Criterion 1 – Original Research...",
      "response": {
        "criterion_1": {
          "satisfied": true,
          "reason": "This is an original research article..."
        }
      },
      "raw_response": "{\"criterion_1\": {...}}"
    }
  ],
  "final_classification": {
    "criteria": {...},
    "final_classification": "relevant",
    "justification": "Overall reasoning..."
  },
  "chunks_processed": 15,
  "errors": []
}
```

## Criteria Evaluated

The script evaluates papers on 6 criteria:

1. **Original Research**: Is it an original research article?
2. **AD Focus**: Is Alzheimer's Disease the main focus?
3. **Sample Size**: Does the human study have n >= 50?
4. **Protein Biomarkers**: Does it focus on protein biomarkers?
5. **Animal Models Exclusion**: Does it use human data only?
6. **Blood as AD Biomarker**: Is blood used as an AD biomarker?

Plus a final classification that aggregates all criteria.

## Error Handling

- **Queue safety**: Uses claim/ack pattern to prevent data loss
- **Requeuing**: Failed papers are requeued for retry
- **Intermediate saves**: Results are saved incrementally
- **Provider fallback**: Continues processing even if some providers fail

## File Outputs

- `rag_labeling_results_YYYYMMDD_HHMMSS.json`: Timestamped results
- `intermediate_results_N.json`: Intermediate saves during processing
- `final_rag_labeling_results.json`: Final results file

## Troubleshooting

1. **No papers in queue**: Make sure to seed the queue first
2. **API key errors**: Check that your API keys are set correctly
3. **Vector store errors**: Ensure Pinecone is configured and the namespace exists
4. **Redis connection errors**: Make sure Redis is running and accessible

## Performance Tips

- Use lower temperature settings (0.1) for more consistent results
- Process papers in smaller batches to avoid memory issues
- Monitor API rate limits for your providers
- Use intermediate saves to prevent data loss on long runs
