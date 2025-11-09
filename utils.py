import re


def clean_json_text(text: str) -> str:
	"""Attempt to extract and minimally clean JSON from LLM output."""
	text = text.strip()
	match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
	if match:
		candidate = match.group(1)
		candidate = re.sub(r",\s*(\]|\})", r"\1", candidate)
		return candidate
	return text


