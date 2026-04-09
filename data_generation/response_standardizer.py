#!/usr/bin/env python3
"""
Response Standardizer for LLM Provider Outputs

This module provides functions to standardize the output formats from different
LLM providers to ensure consistent JSON parsing across all providers.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple


def standardize_llm_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """
    Standardize LLM response to ensure consistent JSON parsing.
    
    Args:
        raw_response: The raw response string from the LLM provider
        
    Returns:
        Tuple of (parsed_json, cleaned_content, success_flag)
        - parsed_json: The parsed JSON object if successful, None otherwise
        - cleaned_content: The cleaned JSON string
        - success_flag: True if parsing was successful, False otherwise
    """
    
    if not raw_response or not raw_response.strip():
        return None, "", False
    
    # Step 1: Clean the response content
    cleaned_content = clean_response_content(raw_response.strip())
    
    # Step 2: Try to parse the cleaned content
    try:
        parsed_json = json.loads(cleaned_content)
        return parsed_json, cleaned_content, True
    except json.JSONDecodeError as e:
        # Step 3: If parsing fails, try additional cleaning strategies
        cleaned_content = apply_additional_cleaning(cleaned_content)
        try:
            parsed_json = json.loads(cleaned_content)
            return parsed_json, cleaned_content, True
        except json.JSONDecodeError:
            return None, cleaned_content, False


def clean_response_content(content: str) -> str:
    """
    Clean response content by removing markdown formatting and other artifacts.
    
    Args:
        content: Raw response content
        
    Returns:
        Cleaned content string
    """
    
    # Remove markdown code blocks
    if content.startswith('```json') and content.endswith('```'):
        # Remove ```json and ``` markers
        content = content[7:-3].strip()
    elif content.startswith('```') and content.endswith('```'):
        # Remove generic ``` markers
        content = content[3:-3].strip()
    
    # Remove any remaining markdown artifacts
    content = content.strip()
    
    return content


def apply_additional_cleaning(content: str) -> str:
    """
    Apply additional cleaning strategies for stubborn responses.
    
    Args:
        content: Content that failed initial parsing
        
    Returns:
        Further cleaned content
    """
    
    # Remove any leading/trailing whitespace
    content = content.strip()
    
    # Try to find JSON within the content if it's embedded in other text
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, content, re.DOTALL)
    if json_match:
        content = json_match.group(0)
    
    # Remove any remaining non-JSON artifacts
    content = content.strip()
    
    return content


def validate_json_structure(parsed_json: Dict[str, Any], expected_keys: list = None) -> bool:
    """
    Validate that the parsed JSON has the expected structure.
    
    Args:
        parsed_json: The parsed JSON object
        expected_keys: List of expected top-level keys
        
    Returns:
        True if structure is valid, False otherwise
    """
    
    if not isinstance(parsed_json, dict):
        return False
    
    if expected_keys:
        for key in expected_keys:
            if key not in parsed_json:
                return False
    
    return True


def extract_criterion_result(parsed_json: Dict[str, Any], criterion_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract the result for a specific criterion from the parsed JSON.
    
    Args:
        parsed_json: The parsed JSON object
        criterion_name: The name of the criterion (e.g., "criterion_1")
        
    Returns:
        The criterion result if found, None otherwise
    """
    
    if not isinstance(parsed_json, dict):
        return None
    
    return parsed_json.get(criterion_name)


def create_standardized_result(
    criterion: str,
    prompt: str,
    raw_response: str,
    chunks_used: int,
    provider: str = None
) -> Dict[str, Any]:
    """
    Create a standardized result structure for a criterion evaluation.
    
    Args:
        criterion: The criterion name (e.g., "criterion_1")
        prompt: The prompt used for this criterion
        raw_response: The raw response from the LLM
        chunks_used: Number of chunks used
        provider: The provider name (optional)
        
    Returns:
        Standardized result dictionary
    """
    
    # Standardize the response
    parsed_json, cleaned_content, success = standardize_llm_response(raw_response)
    
    # Create the result structure
    result = {
        "criterion": criterion,
        "prompt": prompt,
        "raw_response": raw_response,
        "chunks_used": chunks_used
    }
    
    if provider:
        result["provider"] = provider
    
    if success and parsed_json:
        result["response"] = parsed_json
        result["cleaned_response"] = cleaned_content
    else:
        result["response"] = None
        result["error"] = "Failed to parse JSON"
        result["cleaned_response"] = cleaned_content
    
    return result


def test_standardizer():
    """
    Test the standardizer with various response formats.
    """
    
    test_cases = [
        # Case 1: JSON with ```json code blocks
        '```json\n{"criterion_1": {"satisfied": true, "reason": "Test reason"}}\n```',
        
        # Case 2: JSON with generic ``` code blocks
        '```\n{"criterion_2": {"satisfied": false, "reason": "Test reason"}}\n```',
        
        # Case 3: Plain JSON
        '{"criterion_3": {"satisfied": true, "reason": "Test reason"}}',
        
        # Case 4: Multi-line JSON with code blocks
        '''```json
{
  "criterion_4": {
    "satisfied": false,
    "reason": "Multi-line test reason"
  }
}
```''',
        
        # Case 5: JSON with extra text
        'Here is the result: {"criterion_5": {"satisfied": true, "reason": "Test reason"}}',
        
        # Case 6: Invalid JSON
        'This is not valid JSON at all'
    ]
    
    print("Testing Response Standardizer")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {repr(test_case)}")
        
        parsed_json, cleaned_content, success = standardize_llm_response(test_case)
        
        print(f"Success: {success}")
        print(f"Cleaned: {repr(cleaned_content)}")
        if success:
            print(f"Parsed: {parsed_json}")
        else:
            print("Failed to parse JSON")


if __name__ == "__main__":
    test_standardizer()
