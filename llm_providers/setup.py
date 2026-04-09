from setuptools import setup, find_packages

setup(
    name="llm_providers",
    version="0.1.0",
    description="Unified interface for different LLM providers",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-huggingface>=0.1.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    python_requires=">=3.8",
)
