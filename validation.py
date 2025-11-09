from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from urllib.parse import urlparse
import re
import hashlib
import difflib
from diskcache import Cache
import logging
import requests
from bs4 import BeautifulSoup
import json
from utils import clean_json_text

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
	"name",
	"url",
	"exchange",
	"ticker",
	"business_activity",
	"customer_segment",
	"SIC_industry",
]


def _has_required_fields(entry: Dict[str, Any]) -> bool:
	if not isinstance(entry, dict):
		return False
	for k in REQUIRED_KEYS:
		if k not in entry:
			return False
		v = entry[k]
		if v is None:
			return False
		if isinstance(v, str):
			if not v.strip():
				return False
			# Allow 'unknown' values (no strict filtering)
	return True


def validate_and_normalize_output(data: Dict[str, Any]) -> Dict[str, Any]:
	if not isinstance(data, dict) or "comparables" not in data:
		raise ValueError("Output must be a JSON object with key 'comparables'.")
	comparables = data.get("comparables")
	if not isinstance(comparables, list):
		raise ValueError("'comparables' must be a list.")

	cleaned: List[Dict[str, Any]] = []
	dropped = 0
	for x in comparables:
		if not _has_required_fields(x):
			dropped += 1
			continue
		cleaned.append({k: (x.get(k) or "").strip() if isinstance(x.get(k), str) else x.get(k) for k in REQUIRED_KEYS})

	if len(cleaned) == 0:
		logger.error("Validation failed: received=%s dropped=%s", len(comparables), dropped)
		raise ValueError("No valid comparable entries with all required fields.")

	logger.info("Validated comparables: received=%s cleaned=%s dropped=%s", len(comparables), len(cleaned), dropped)
	return {"comparables": cleaned}


_EMB_CACHE = Cache("emb_cache")
_SEC_PROFILE_CACHE = Cache("sec_profile_cache")
_US_EXCHANGE_KEYWORDS = (
	"NYSE",
	"NASDAQ",
	"ARCA",
	"AMEX",
	"CBOE",
	"IEX",
	"OTC",
	"BATS",
)
def _hash_id(text: str, model: str) -> str:
	return f"{model}:" + hashlib.sha256((model + '::' + (text or '')).encode('utf-8')).hexdigest()

def _get_embeddings_cached(texts: List[str], openai_api_key: str, model: str) -> List[List[float]]:
	# Embedding-based similarity disabled; return empty to trigger fallback
	return []


def _compute_embeddings(texts: List[str], openai_api_key: str) -> List[List[float]]:
	return []

def _keyword_overlap(a: str, b: str) -> float:
	a_set = {t.lower() for t in a.split() if t.isalpha() or t.isalnum()}
	b_set = {t.lower() for t in b.split() if t.isalpha() or t.isalnum()}
	if not a_set or not b_set:
		return 0.0
	overlap = len(a_set & b_set)
	return overlap / float(min(len(a_set), len(b_set)))


def _simple_extract_keywords(text: str, max_tokens: int = 12) -> List[str]:
	stop = {
		"the","and","for","with","that","this","from","into","their","its","are","was",
		"of","in","to","on","a","an","by","as","at","or","be","is","it","we","our","they",
	}
	words: List[str] = []
	for raw in (text or "").replace("/", " ").replace("-", " ").split():
		w = ''.join(ch for ch in raw.lower() if ch.isalnum())
		if not w or len(w) < 3 or w in stop:
			continue
		words.append(w)
	# Preserve order of first occurrence, then truncate
	seen = set()
	uniq: List[str] = []
	for w in words:
		if w in seen:
			continue
		seen.add(w)
		uniq.append(w)
	return uniq[:max_tokens]


def _build_candidate_text(c: Dict[str, Any]) -> str:
	parts = [
		c.get("name") or "",
		c.get("business_activity") or "",
		c.get("customer_segment") or "",
		c.get("SIC_industry") or "",
	]
	url = c.get("url") or ""
	try:
		domain = urlparse(url).netloc
		if domain:
			parts.append(domain)
	except Exception:
		pass
	return " \n".join(p for p in parts if p)


def _min_max_scale(values: List[float]) -> List[float]:
	if not values:
		return values
	vmin = min(values)
	vmax = max(values)
	if vmax - vmin == 0:
		return [0.5 for _ in values]
	return [(v - vmin) / (vmax - vmin) for v in values]


def rank_and_filter_by_similarity(
	target_description: str,
	comparables: List[Dict[str, Any]],
	openai_api_key: str,
	min_results: int = 3,
	max_results: int = 10,
	threshold: float = 0.55,
	desc_weight: float = 0.6,
	kw_weight: float = 0.4,
) -> Dict[str, Any]:
	"""Compute multi-feature similarity and attach similarity_score to each comparable.

	Score = 0.6*cos(desc_emb, cand_emb) + 0.4*cos(keywords_emb, cand_emb)
	"""
	# Prepare texts
	keywords = _simple_extract_keywords(target_description)
	kw_text = " ".join(keywords)
	candidate_texts = [_build_candidate_text(c) for c in comparables]
	texts = [target_description, kw_text] + candidate_texts
	# Embeddings disabled: keyword overlap only
	scored = []
	for comp in comparables:
		cand_text = comp.get("business_activity", "")
		score = _keyword_overlap(target_description, cand_text)
		scored.append((score, comp))

	# Calibrate with min-max scaling across cohort and rank internally (no score field attached)
	values = [s for s, _ in scored]
	scaled = _min_max_scale(values)
	# Rank using scaled scores
	ranked = list(zip(scaled, [c for _, c in scored]))
	ranked.sort(key=lambda x: x[0], reverse=True)
	filtered = [c for s, c in ranked if s >= threshold]
	if len(filtered) < min_results:
		filtered = [c for _, c in ranked][:max_results]
	else:
		filtered = filtered[:max_results]
	return {"comparables": filtered}


def add_reasoning_notes(target_description: str, target_name: str, comparables: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Attach a short, human-friendly reasoning sentence to each comparable.

	Deterministic template-based explanation referencing industry, activity, and segment.
	"""
	updated: List[Dict[str, Any]] = []
	for c in comparables:
		industry = c.get("SIC_industry") or "similar industries"
		activity = c.get("business_activity") or "related offerings"
		segment = c.get("customer_segment") or "similar customers"
		tgt_kws = set(_simple_extract_keywords(target_description))
		cand_kws = _token_set(" ".join([industry, activity, segment]))
		matched = sorted(list(tgt_kws & cand_kws))[:2]
		match_note = f" Matched keywords: {', '.join(matched)}." if matched else ""
		score_txt = f"Score: {c.get('similarity_score')} — " if "similarity_score" in c else ""
		reason = (
			f"{score_txt}operates in {industry}, offering {activity} to {segment}.{match_note}"
		)
		new_c = dict(c)
		new_c["reasoning"] = reason
		updated.append(new_c)
	return {"comparables": updated}


# --- SEC SIC list scraping & resolution ---
def _sec_headers() -> Dict[str, str]:
	return {
		"User-Agent": "ComparableFinder/1.0 (contact: example@example.com)",
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
		"Connection": "keep-alive",
		"Referer": "https://www.sec.gov/",
	}


def fetch_sic_index() -> List[Dict[str, str]]:
	"""Scrape SEC SIC code list and return [{'code': '####', 'name': 'Industry'}]."""
	url = "https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list"
	r = requests.get(url, headers=_sec_headers(), timeout=20)
	r.raise_for_status()
	html = r.text or ""
	soup = BeautifulSoup(html, "lxml")
	entries: List[Dict[str, str]] = []
	for table in soup.find_all("table"):
		for tr in table.find_all("tr"):
			tds = tr.find_all(["td", "th"])
			if len(tds) < 2:
				continue
			code_txt = (tds[0].get_text(strip=True) or "")
			name_txt = (tds[1].get_text(" ", strip=True) or "")
			m = re.match(r"^\s*(\d{4})\s*$", code_txt)
			if m and name_txt:
				entries.append({"code": m.group(1), "name": name_txt})
	if not entries:
		for m in re.finditer(r"\b(\d{4})\b\s*[-:–]\s*([A-Za-z][^\n\r<]+)", html):
			entries.append({"code": m.group(1), "name": m.group(2).strip()})
	seen = set()
	deduped: List[Dict[str, str]] = []
	for e in entries:
		if e["code"] in seen:
			continue
		seen.add(e["code"])
		deduped.append(e)
	logger.info("SEC SIC index parsed=%s deduped=%s", len(entries), len(deduped))
	return deduped


def _norm_tokens(text: str) -> List[str]:
	words: List[str] = []
	for raw in (text or "").replace("/", " ").replace("-", " ").split():
		w = ''.join(ch for ch in raw.lower() if ch.isalnum())
		if not w or len(w) < 2:
			continue
		words.append(w)
	seen = set()
	out: List[str] = []
	for w in words:
		if w in seen:
			continue
		seen.add(w)
		out.append(w)
	return out


def _token_overlap(a: List[str], b: List[str]) -> float:
	if not a or not b:
		return 0.0
	aset, bset = set(a), set(b)
	if not aset or not bset:
		return 0.0
	inter = len(aset & bset)
	return inter / float(min(len(aset), len(bset)))


def _norm_text(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _is_us_exchange(exchange: str) -> bool:
	ex = (exchange or "").upper()
	if not ex:
		return False
	return any(keyword in ex for keyword in _US_EXCHANGE_KEYWORDS)


def resolve_sic(user_input: str) -> Dict[str, Any]:
	"""Resolve a user-provided SIC code or industry text to the closest SEC SIC entry using only the user input."""
	entries = fetch_sic_index()
	ui = (user_input or "").strip()
	logger.info("resolve_sic: raw_input='%s'", ui)
	if not ui:
		logger.error("resolve_sic: empty user input")
		raise ValueError("SIC input is required.")
	if re.fullmatch(r"\d{4}", ui):
		logger.info("resolve_sic: numeric input detected")
		for e in entries:
			if e["code"] == ui:
				logger.info("resolve_sic: exact code match %s -> %s", ui, e["name"])
				return {"match": {"code": e["code"], "name": e["name"], "score": 1.0, "method": "exact_code"}, "candidates": [e]}
	src_tokens = _norm_tokens(ui)
	ui_text = _norm_text(ui)
	logger.info("resolve_sic: tokens=%s norm_text='%s'", src_tokens, ui_text)
	# Exact text match (normalized)
	for e in entries:
		if _norm_text(e["name"]) == ui_text and ui_text:
			logger.info("resolve_sic: exact name match %s -> %s", ui, e["code"])
			return {"match": {"code": e["code"], "name": e["name"], "score": 1.0, "method": "exact_name"}, "candidates": [e]}
	# Substring matches based on multi-word overlap
	sub_matches = []
	if ui_text:
		ui_parts = ui_text.split()
		for e in entries:
			name_norm = _norm_text(e["name"])
			if all(part in name_norm for part in ui_parts):
				sub_matches.append(e)
	if sub_matches:
		best_sub = sub_matches[0]
		logger.info("resolve_sic: substring match %s -> %s", ui, best_sub["code"])
		return {"match": {"code": best_sub["code"], "name": best_sub["name"], "score": 0.95, "method": "substring"}, "candidates": sub_matches[:5]}
	best_score = -1.0
	best_entry: Dict[str, str] | None = None
	best_method = "name"
	scored: List[Tuple[float, Dict[str, str]]] = []
	for e in entries:
		name_text = _norm_text(e["name"])
		tokens = _norm_tokens(e["name"])
		score_name_tokens = _token_overlap(src_tokens, tokens) if src_tokens else 0.0
		score_name_ratio = difflib.SequenceMatcher(None, ui_text, name_text).ratio() if ui_text else 0.0
		score = max(score_name_tokens, score_name_ratio)
		scored.append((score, e))
		if score > best_score:
			best_score = score
			best_entry = e
			best_method = "tokens" if score_name_tokens >= score_name_ratio else "ratio"
	scored.sort(key=lambda x: x[0], reverse=True)
	top = [e for _, e in scored[:5]]
	if best_entry is None or best_score <= 0:
		logger.error("resolve_sic: unable to map input '%s' to any SIC entry", ui)
		raise ValueError(f"Unable to resolve SIC from input: {ui}")
	logger.info("resolve_sic: fuzzy match %s -> %s (score=%.3f, method=%s)", ui, best_entry["code"], best_score, best_method)
	return {"match": {"code": best_entry["code"], "name": best_entry["name"], "score": round(float(best_score), 4), "method": best_method}, "candidates": top}


def _parse_sec_browse_page(html: str) -> List[Dict[str, str]]:
	soup = BeautifulSoup(html or "", "lxml")
	results: List[Dict[str, str]] = []
	table = soup.find("table", class_="tableFile2")
	if table:
		rows = table.find_all("tr")
		for tr in rows[1:]:
			tds = tr.find_all("td")
			if len(tds) < 2:
				continue
			cik_text = tds[0].get_text(strip=True) or ""
			company_text = tds[1].get_text(" ", strip=True) or ""
			if not cik_text or not company_text:
				continue
			a = tds[0].find("a", href=True) or tds[1].find("a", href=True)
			url = ""
			if a:
				href = a["href"]
				url = href if href.startswith("http") else ("https://www.sec.gov" + href)
			results.append({"name": company_text, "cik": cik_text, "url": url})
	if not results:
		for a in soup.find_all("a", href=True):
			href = a["href"]
			if "action=getcompany" in href and "CIK=" in href:
				name = (a.get_text(" ", strip=True) or "").strip()
				m = re.search(r"CIK=(\d+)", href)
				cik = m.group(1) if m else ""
				if not name or not cik:
					continue
				url = href if href.startswith("http") else ("https://www.sec.gov" + href)
				results.append({"name": name, "cik": cik, "url": url})
	return results


def fetch_company_profile_from_sec_submissions(cik: str) -> Dict[str, str]:
	cik_clean = (cik or "").strip()
	if not cik_clean:
		return {}
	cache_key = f"sec_submissions:{cik_clean}"
	cached = _SEC_PROFILE_CACHE.get(cache_key)
	if cached is not None:
		return cached
	url = f"https://data.sec.gov/submissions/CIK{cik_clean.zfill(10)}.json"
	try:
		resp = requests.get(url, headers=_sec_headers(), timeout=20)
		if not resp.ok:
			logger.warning("SEC submissions: %s responded %s", url, resp.status_code)
			profile = {}
		else:
			data = resp.json() or {}
			tickers_raw = data.get("tickers") or []
			exchanges_raw = data.get("exchanges") or []
			tickers = [str(t or "").strip() for t in tickers_raw if t]
			exchanges = [str(e or "").strip() for e in exchanges_raw if e]
			website = ""
			if isinstance(data.get("website"), str):
				website = data["website"].strip()
			elif isinstance(data.get("entity"), dict):
				website = str(data["entity"].get("website") or "").strip()
			if not website:
				entity_info = data.get("entityInfo") or data.get("entity") or {}
				if isinstance(entity_info, dict):
					website = str(entity_info.get("website") or "").strip()
			profile = {
				"ticker": tickers[0] if tickers else "",
				"exchange": exchanges[0] if exchanges else "",
				"business_url": website,
			}
	except Exception:
		logger.exception("fetch_company_profile_from_sec_submissions failed for CIK %s", cik_clean)
		profile = {}
	_SEC_PROFILE_CACHE.set(cache_key, profile, expire=12 * 3600)
	return profile


def fetch_companies_by_sic(
	sic_code: str,
	start: int = 0,
	count: int = 40,
	owner: str = "include",
) -> List[Dict[str, str]]:
	"""Scrape SEC browse-EDGAR pages for a given SIC and return company info (with pagination)."""
	base = "https://www.sec.gov/cgi-bin/browse-edgar"
	all_results: List[Dict[str, str]] = []
	page = 0
	offset = start
	while True:
		params = {
			"action": "getcompany",
			"SIC": str(sic_code),
			"owner": owner,
			"match": "starts-with",
			"start": str(offset),
			"count": str(count),
			"hidefilings": "0",
		}
		logger.info("SEC scrape page=%s SIC=%s offset=%s count=%s owner=%s", page, sic_code, offset, count, owner)
		r = requests.get(base, params=params, headers=_sec_headers(), timeout=20)
		r.raise_for_status()
		page_results = _parse_sec_browse_page(r.text or "")
		logger.info("SEC scrape page=%s results=%s", page, len(page_results))
		if not page_results:
			break
		all_results.extend(page_results)
		if len(page_results) < count:
			break
		page += 1
		offset += count
	# Deduplicate by CIK while preserving order
	seen = set()
	deduped: List[Dict[str, str]] = []
	for e in all_results:
		cik = e.get("cik") or ""
		if cik in seen:
			continue
		seen.add(cik)
		deduped.append(e)
	for entry in deduped:
		entry.setdefault("name", "")
		entry.setdefault("url", "")
		entry.setdefault("cik", "")
		entry.setdefault("ticker", "")
		entry.setdefault("exchange", "")
	with ThreadPoolExecutor(max_workers=8) as executor:
		futures = {
			executor.submit(fetch_company_profile_from_sec_submissions, entry.get("cik") or ""): entry
			for entry in deduped
			if entry.get("cik")
		}
		for future in as_completed(futures):
			entry = futures[future]
			try:
				profile = future.result()
			except Exception:
				logger.exception("SEC submissions enrichment failed for CIK %s", entry.get("cik"))
				profile = {}
			if not profile:
				continue
			if profile.get("ticker"):
				entry["ticker"] = profile["ticker"]
			if profile.get("exchange"):
				entry["exchange"] = profile["exchange"]
			if profile.get("business_url"):
				entry["url"] = profile["business_url"]
	filtered = [
		entry for entry in deduped
		if (entry.get("ticker") or "").strip() and (entry.get("exchange") or "").strip()
	]
	dropped = len(deduped) - len(filtered)
	if dropped:
		logger.info("Removed %s companies missing ticker or exchange after SEC enrichment", dropped)
	us_filtered = [entry for entry in filtered if _is_us_exchange(entry.get("exchange"))]
	us_dropped = len(filtered) - len(us_filtered)
	if us_dropped:
		logger.info("Removed %s companies not trading on US exchanges", us_dropped)
	deduped = us_filtered
	logger.info("SEC scrape complete: SIC=%s total_parsed=%s unique=%s pages=%s", sic_code, len(all_results), len(deduped), page + 1)
	for e in deduped[:10]:
		logger.debug("SEC company: name=%s cik=%s url=%s ticker=%s exchange=%s", e.get("name"), e.get("cik"), e.get("url"), e.get("ticker"), e.get("exchange"))
	return deduped


def filter_companies_with_llm(
	target_name: str,
	target_description: str,
	companies: List[Dict[str, Any]],
	openai_api_key: str,
	max_companies: int = 80,
) -> List[Dict[str, Any]]:
	if not companies or not openai_api_key:
		return companies
	client = OpenAI(api_key=openai_api_key)
	system_prompt = (
		"You are an equity research analyst. KEEP only U.S.-listed" 
		"public peers/competitors of the target (NYSE, NASDAQ, AMEX) with clearly "
		"overlapping services, customer segments, and business model."
		"DROP non-U.S. listings, private firms, ADRs/ETFs, shells/holdcos,"
		"suppliers/customers, and companies with materially different models."
	)
	filtered_all: List[Dict[str, Any]] = []
	for offset in range(0, len(companies), max_companies):
		subset = companies[offset : offset + max_companies]
		payload = {
			"target": {
				"name": target_name or "",
				"description": target_description or "",
			},
			"candidates": [
				{
					"index": idx,
					"name": comp.get("name") or "",
					"ticker": comp.get("ticker") or "",
					"exchange": comp.get("exchange") or "",
					"url": comp.get("url") or "",
					"sic": comp.get("SIC_industry") or "",
				}
				for idx, comp in enumerate(subset)
			],
		}
		user_prompt = (
			"Review the target and candidate companies below. Return JSON with a single key 'keep' whose value "
			"is a list of candidate indexes to keep. Remove any company that is not a clear US comparable. "
			"Return strictly valid JSON. No explanations.\n\n"
			f"{json.dumps(payload, ensure_ascii=False)}"
		)
		try:
			resp = client.chat.completions.create(
				model="gpt-4o",
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				max_tokens=600,
			)
			text = resp.choices[0].message.content if resp.choices else "{}"
			data = json.loads(clean_json_text(text))
			keep_raw = data.get("keep") or data.get("include") or []
			keep_indices = set()
			for item in keep_raw:
				try:
					keep_indices.add(int(item))
				except Exception:
					continue
			filtered_subset = [
				company
				for idx, company in enumerate(subset)
				if idx in keep_indices
			]
			if not filtered_subset:
				logger.info("LLM filtering removed all %s candidates in batch starting at %s", len(subset), offset)
			filtered_all.extend(filtered_subset)
		except Exception:
			logger.exception("filter_companies_with_llm failed for batch starting at %s", offset)
			filtered_all.extend(subset)
	logger.info("LLM filtering kept %s of %s candidates", len(filtered_all), len(companies))
	return filtered_all


def map_sic_with_llm(user_input: str, openai_api_key: str) -> Dict[str, Any]:
	"""Use OpenAI LLM to map free-text SIC description to a 4-digit SIC code."""
	if not openai_api_key:
		raise ValueError("OpenAI API key is required for SIC mapping.")
	client = OpenAI(api_key=openai_api_key)
	instruction = (
		"You are an expert on SEC Standard Industrial Classification (SIC) codes. "
		"Given a short description, Primary SIC Industry Classification, return the single best matching 4-digit SIC code. "
		"Respond STRICTLY in JSON with fields: sic_code (string, 4 digits), sic_name (official description), confidence (float between 0 and 1). "
	)
	logger.info("map_sic_with_llm: querying LLM with input '%s'", user_input)
	resp = client.chat.completions.create(
		model="gpt-5",
		messages=[
			{"role": "system", "content": instruction},
			{"role": "user", "content": f"Industry description: {user_input.strip()}"},
		],
	)
	text = resp.choices[0].message.content if resp.choices else "{}"
	try:
		data = json.loads(clean_json_text(text))
	except Exception as exc:
		logger.exception("map_sic_with_llm: failed to parse LLM response '%s'", text)
		raise ValueError(f"OpenAI response parsing failed: {exc}") from exc
	code = str(data.get("sic_code") or "").strip()
	name = (data.get("sic_name") or "").strip()
	conf = data.get("confidence")
	if not re.fullmatch(r"\d{4}", code):
		logger.error("map_sic_with_llm: invalid sic_code '%s' from LLM response '%s'", code, text)
		raise ValueError(f"LLM did not return a valid 4-digit SIC code for input '{user_input}'.")
	logger.info("map_sic_with_llm: LLM mapped '%s' -> %s (%s) confidence=%s", user_input, code, name, conf)
	return {"sic_code": code, "sic_name": name, "confidence": conf}

def _token_set(text: str) -> set:
	return {t.lower() for t in (text or "").split() if t.isalpha() or t.isalnum()}


def _name_matches(target_name: str, candidate_name: str, threshold: float = 0.5) -> bool:
	if not target_name or not candidate_name:
		return False
	t = _token_set(target_name)
	c = _token_set(candidate_name)
	if not t or not c:
		return False
	inter = len(t & c)
	return (inter / float(min(len(t), len(c)))) >= threshold


# --- Similarity scoring (embeddings + optional enrichment) ---

def _cosine_similarity(a: List[float], b: List[float]) -> float:
	va = np.array(a)
	vb = np.array(b)
	den = (np.linalg.norm(va) * np.linalg.norm(vb))
	if den == 0:
		return 0.0
	return float(np.dot(va, vb) / den)


def _comp_similarity_text(comp: Dict[str, Any]) -> str:
	parts = [
		comp.get("name") or "",
		comp.get("business_activity") or "",
		comp.get("customer_segment") or "",
		comp.get("SIC_industry") or "",
	]
	desc = " | ".join(p for p in parts if p)
	if len(desc.split()) < 20:
		desc += " | Provides offerings to defined customer segments in related end-markets."
	return desc


def compute_similarity_scores(
	target_description: str,
	comparables: List[Dict[str, Any]],
	openai_api_key: str,
) -> Dict[str, Any]:
	texts = [target_description] + [_comp_similarity_text(c) for c in comparables]
	# Embeddings disabled: keyword overlap using business_activity
	updated = []
	for comp in comparables:
		cand = comp.get("business_activity", "")
		score = _keyword_overlap(target_description, cand)
		new_c = dict(comp)
		new_c["similarity_score"] = round(float(score), 4)
		updated.append(new_c)
	return {"comparables": updated}


def enrich_similarity_scores(
	target_description: str,
	target_sic: str,
	comparables: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Combine embedding score with keyword overlap and SIC industry bonus."""
	t_tokens = _token_set(target_description)
	updated: List[Dict[str, Any]] = []
	for comp in comparables:
		base = float(comp.get("similarity_score", 0.0))
		cand_text = " ".join([
			comp.get("business_activity") or "",
			comp.get("customer_segment") or "",
			comp.get("SIC_industry") or "",
		])
		overlap = 0.0
		c_tokens = _token_set(cand_text)
		if t_tokens and c_tokens:
			overlap = len(t_tokens & c_tokens) / float(max(len(t_tokens), 1))
		industry_bonus = 0.0
		if target_sic and (target_sic in (comp.get("SIC_industry") or "")):
			industry_bonus = 0.1
		# weights sum to 1.0
		w_embed = 0.75
		w_kw = 0.15
		w_bonus = 0.10
		weighted = w_embed * base + w_kw * overlap + w_bonus * industry_bonus
		new_c = dict(comp)
		new_c["similarity_score"] = round(max(0.0, min(1.0, weighted)), 4)
		updated.append(new_c)
	return {"comparables": updated}


def normalize_similarity_scores(comparables: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Leave scores as-is (already in [0,1]); clamp to bounds without forcing extremes."""
	updated: List[Dict[str, Any]] = []
	for c in comparables:
		val = c.get("similarity_score")
		if isinstance(val, (int, float)):
			clamped = max(0.0, min(1.0, float(val)))
			new_c = dict(c)
			new_c["similarity_score"] = round(clamped, 4)
			updated.append(new_c)
		else:
			updated.append(c)
	return {"comparables": updated}


