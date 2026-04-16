from typing import Dict


def health_check_handler() -> Dict[str, str]:
    return {"status": "healthy"}
