import json
import random
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 0.8, jitter: float = 0.2) -> Callable[[F], F]:
	def decorator(func: F) -> F:
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			for attempt in range(max_retries + 1):
				try:
					return func(*args, **kwargs)
				except Exception as e:
					if attempt >= max_retries:
						raise
					delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
					time.sleep(delay)
		return wrapper  # type: ignore

	return decorator


def clean_json_text(text: str) -> str:
	"""Attempt to extract and minimally clean JSON from LLM output."""
	text = text.strip()
	# Extract first {...} or [...]
	match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
	if match:
		candidate = match.group(1)
		# Remove trailing commas before } or ]
		candidate = re.sub(r",\s*(\]|\})", r"\1", candidate)
		return candidate
	return text


def unique_by_key(items: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> List[Dict[str, Any]]:
	seen = set()
	result: List[Dict[str, Any]] = []
	for item in items:
		k = key_fn(item)
		if k in seen:
			continue
		seen.add(k)
		result.append(item)
	return result


