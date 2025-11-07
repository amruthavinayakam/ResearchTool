from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI
import requests
from urllib.parse import quote, urlparse
import re
from bs4 import BeautifulSoup
import functools
import hashlib
from diskcache import Cache
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv
import io
import os

logger = logging.getLogger(__name__)

# --- Yahoo session (retry + headers) ---
_YAHOO_SESSION: requests.Session | None = None


def _yahoo_session() -> requests.Session:
	global _YAHOO_SESSION
	if _YAHOO_SESSION is None:
		s = requests.Session()
		retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
		adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
		s.mount("https://", adapter)
		s.mount("http://", adapter)
		_YAHOO_SESSION = s
	return _YAHOO_SESSION


def _yahoo_headers() -> Dict[str, str]:
	return {
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
		"Accept": "application/json,text/plain,*/*",
		"Connection": "keep-alive",
	}


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
_AV_CACHE = Cache("av_cache")

# --- Alpha Vantage session (retry) ---
_AV_SESSION: requests.Session | None = None


def _av_session() -> requests.Session:
	global _AV_SESSION
	if _AV_SESSION is None:
		s = requests.Session()
		retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
		adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
		s.mount("https://", adapter)
		s.mount("http://", adapter)
		_AV_SESSION = s
	return _AV_SESSION


def _av_listing_status(api_key: str, allowed_exchanges: List[str]) -> Dict[str, str]:
	"""Return mapping ticker -> exchange for active listings limited to allowed_exchanges.

	Results cached on disk for 24 hours.
	"""
	key = f"listing_status:{','.join(sorted(allowed_exchanges)).upper()}"
	row = _AV_CACHE.get(key, default=None)
	if row is not None:
		return row
	url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&state=active&apikey={api_key}"
	r = _av_session().get(url, timeout=30)
	r.raise_for_status()
	content = r.text
	buf = io.StringIO(content)
	reader = csv.DictReader(buf)
	allow = {e.upper() for e in allowed_exchanges}
	mapping: Dict[str, str] = {}
	for rec in reader:
		symbol = (rec.get("symbol") or "").strip()
		exchange = (rec.get("exchange") or "").strip().upper()
		status = (rec.get("status") or "").strip().lower()
		if not symbol or not exchange:
			continue
		if exchange not in allow:
			continue
		if status and status != "active":
			continue
		mapping[symbol.upper()] = exchange
	# cache for 24h
	_AV_CACHE.set(key, mapping, expire=24 * 3600)
	return mapping


def filter_us_active_listed(
	comparables: List[Dict[str, Any]],
	alpha_vantage_api_key: str,
	allowed_exchanges: List[str] | None = None,
) -> Dict[str, Any]:
	"""Keep only U.S. active listings on allowed U.S. exchanges using Alpha Vantage LISTING_STATUS.

	Sets exchange from AV mapping when present.
	"""
	allowed = allowed_exchanges or ["NYSE", "NASDAQ", "AMEX"]
	if not alpha_vantage_api_key:
		# If no key, conservatively return empty list to avoid mislabeling
		logger.warning("Alpha Vantage key missing; cannot filter by active listing")
		return {"comparables": []}
	mapping = _av_listing_status(alpha_vantage_api_key, allowed)
	kept: List[Dict[str, Any]] = []
	for comp in comparables:
		sym = (comp.get("ticker") or comp.get("symbol") or "").strip().upper()
		if not sym:
			continue
		exch = mapping.get(sym)
		if not exch:
			continue
		new_c = dict(comp)
		new_c["exchange"] = exch
		kept.append(new_c)
	return {"comparables": kept}

def _hash_id(text: str, model: str) -> str:
	return f"{model}:" + hashlib.sha256((model + '::' + (text or '')).encode('utf-8')).hexdigest()

def _get_embeddings_cached(texts: List[str], openai_api_key: str, model: str) -> List[List[float]]:
	ids = [_hash_id(t or "", model) for t in texts]
	cached: Dict[str, List[float]] = {}
	missing_pairs = []
	for i, t in zip(ids, texts):
		v = _EMB_CACHE.get(i, default=None)
		if v is not None:
			cached[i] = v
		else:
			missing_pairs.append((i, t))
	if missing_pairs:
		logger.info("Embeddings cache: hits=%s misses=%s model=%s", len(cached), len(missing_pairs), model)
		client = OpenAI(api_key=openai_api_key)
		missing_texts = [t for _, t in missing_pairs]
		resp = client.embeddings.create(model=model, input=missing_texts)
		new_vecs = [item.embedding for item in resp.data]
		missing_ids = [i for i, _ in missing_pairs]
		for i, v in zip(missing_ids, new_vecs):
			_EMB_CACHE.set(i, v, expire=7 * 24 * 3600)
			cached[i] = v
	else:
		logger.info("Embeddings cache: all hits=%s model=%s", len(ids), model)
	return [cached[i] for i in ids]


def _compute_embeddings(texts: List[str], openai_api_key: str) -> List[List[float]]:
	return _get_embeddings_cached(texts, openai_api_key, model="text-embedding-3-small")

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
	try:
		embs = _compute_embeddings(texts, openai_api_key)
		desc_vec = embs[0]
		kw_vec = embs[1]
		scored: List[Tuple[float, Dict[str, Any]]] = []
		for idx, comp in enumerate(comparables, start=2):
			cand_vec = embs[idx]
			s1 = (_cosine_similarity(desc_vec, cand_vec) + 1.0) / 2.0
			s2 = (_cosine_similarity(kw_vec, cand_vec) + 1.0) / 2.0
			score = desc_weight * s1 + kw_weight * s2
			# Industry/segment overlap bonus up to +0.15
			tgt_tokens = set(keywords)
			cand_tokens = _token_set((comp.get("SIC_industry") or "") + " " + (comp.get("customer_segment") or ""))
			bonus = min(0.15, 0.03 * len(tgt_tokens & cand_tokens))
			scored.append((max(0.0, min(1.0, score + bonus)), comp))
	except Exception:
		# Fallback: keyword overlap on business_activity
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


@functools.lru_cache(maxsize=2048)
def _yahoo_quote(symbol: str) -> Dict[str, Any]:
	"""Fetch quote details for a symbol from Yahoo Finance. Returns {} if not found."""
	try:
		logger.info("Yahoo quote lookup for %s", symbol)
		url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote(symbol)}"
		r = _yahoo_session().get(url, headers=_yahoo_headers(), timeout=10)
		r.raise_for_status()
		data = r.json() or {}
		results = ((data.get("quoteResponse") or {}).get("result") or [])
		return results[0] if results else {}
	except Exception:
		logger.exception("Yahoo quote failed for %s", symbol)
		return {}


@functools.lru_cache(maxsize=2048)
def _yahoo_autocomplete(query: str) -> List[Dict[str, Any]]:
	"""Use Yahoo autocomplete to search symbols by name. Returns list of candidates."""
	try:
		logger.info("Yahoo autocomplete for %s", query)
		url = f"https://autoc.finance.yahoo.com/autoc?lang=en&region=US&query={quote(query)}"
		r = _yahoo_session().get(url, headers=_yahoo_headers(), timeout=10)
		r.raise_for_status()
		data = r.json() or {}
		return ((data.get("ResultSet") or {}).get("Result") or [])
	except Exception:
		logger.exception("Yahoo autocomplete failed for %s", query)
		return []


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


def _normalize_exchange(quote_obj: Dict[str, Any]) -> str:
	return (
		quote_obj.get("fullExchangeName")
		or quote_obj.get("exchangeDisplay")
		or quote_obj.get("exchange")
		or "unknown"
	)


def ground_ticker_exchange(comparables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ground tickers using SEC company_tickers mapping; leave exchange as-is or 'unknown'.

    Returns {"comparables": updated_list} preserving other fields.
    """

    @functools.lru_cache(maxsize=1)
    def _sec_index_by_ticker() -> Dict[str, Dict[str, Any]]:
        data = _sec_company_tickers()
        idx: Dict[str, Dict[str, Any]] = {}
        try:
            for _, v in (data.items() if isinstance(data, dict) else []):
                t = (v.get("ticker") or "").strip()
                if t:
                    idx[t.lower()] = v
        except Exception:
            pass
        return idx

    def _sec_find_by_name(nm: str) -> Dict[str, Any] | None:
        if not nm:
            return None
        data = _sec_company_tickers()
        best: Tuple[int, Dict[str, Any]] | None = None
        try:
            t_tokens = _token_set(nm)
            for _, v in (data.items() if isinstance(data, dict) else []):
                title = (v.get("title") or "").strip()
                if not title:
                    continue
                score = len(t_tokens & _token_set(title))
                if score > 0 and (best is None or score > best[0]):
                    best = (score, v)
        except Exception:
            return None
        return best[1] if best else None

    idx = _sec_index_by_ticker()
    updated: List[Dict[str, Any]] = []
    changed = 0
    for comp in comparables:
        name = (comp.get("name") or "").strip()
        ticker = (comp.get("ticker") or comp.get("symbol") or "").strip()
        new_comp = dict(comp)
        rec: Dict[str, Any] | None = None
        if ticker:
            rec = idx.get(ticker.lower())
            if not rec:
                rec = _sec_find_by_name(name)
        else:
            rec = _sec_find_by_name(name)
        if rec:
            sec_tkr = (rec.get("ticker") or "").strip()
            if sec_tkr and sec_tkr != new_comp.get("ticker"):
                new_comp["ticker"] = sec_tkr
                changed += 1
        # Keep or set exchange
        new_comp["exchange"] = (new_comp.get("exchange") or "unknown").strip() or "unknown"
        updated.append(new_comp)
    logger.info("Ticker grounding via SEC: processed=%s changed=%s", len(comparables), changed)
    return {"comparables": updated}


def _is_active_equity(q: Dict[str, Any], expected_name: str) -> bool:
	if not q or not q.get("symbol"):
		return False
	if (q.get("quoteType") or "").upper() != "EQUITY":
		return False
	# Require live market price (delisted often lacks this)
	if q.get("regularMarketPrice") is None:
		return False
	qname = q.get("longName") or q.get("shortName") or ""
	return (not expected_name) or _name_matches(expected_name, qname, threshold=0.4)


def filter_listed_public_equities(comparables: List[Dict[str, Any]], min_keep: int = 3) -> Dict[str, Any]:
	"""Keep only active, listed equities (Yahoo quote with price), no fallback."""
	filtered: List[Dict[str, Any]] = []
	for comp in comparables:
		sym = (comp.get("ticker") or "").strip()
		name = comp.get("name") or ""
		if not sym:
			continue
		q = _yahoo_quote(sym)
		if _is_active_equity(q, name):
			filtered.append(comp)
	return {"comparables": filtered}


def filter_us_sec_registrants(comparables: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Keep only U.S. SEC registrants (CIK must resolve by ticker or name)."""
	kept: List[Dict[str, Any]] = []
	for comp in comparables:
		name = (comp.get("name") or "").strip()
		sym = (comp.get("ticker") or comp.get("symbol") or "").strip()
		cik = sec_get_cik_by_ticker(sym) or sec_get_cik_by_name(name)
		if cik:
			kept.append(comp)
	return {"comparables": kept}


# --- Similarity scoring (embeddings + optional enrichment) ---

# --- SEC (EDGAR) enrichment utilities ---

SEC_USER_AGENT = "ComparableFinder/1.0 (Fordham DS; contact: vamrutha.works@gmail.com)"


def _sec_headers() -> Dict[str, str]:
	return {
		"User-Agent": SEC_USER_AGENT,
		"Accept-Encoding": "gzip, deflate",
		"Host": "www.sec.gov",
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
		"Connection": "keep-alive",
		"Referer": "https://www.sec.gov/",
	}


_SEC_SESSION: requests.Session | None = None


def _sec_session() -> requests.Session:
	global _SEC_SESSION
	if _SEC_SESSION is None:
		s = requests.Session()
		retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
		adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
		s.mount("https://", adapter)
		s.mount("http://", adapter)
		_SEC_SESSION = s
	return _SEC_SESSION


@functools.lru_cache(maxsize=1)
def _sec_company_tickers() -> Dict[str, Any]:
	url = "https://www.sec.gov/files/company_tickers.json"
	resp = _sec_session().get(url, headers=_sec_headers(), timeout=20)
	resp.raise_for_status()
	return resp.json() or {}


def sec_get_cik_by_ticker(ticker: str) -> str:
	if not ticker:
		return ""
	data = _sec_company_tickers()
	try:
		for _, v in (data.items() if isinstance(data, dict) else []):
			if (v.get("ticker") or "").lower() == ticker.lower():
				return str(v.get("cik_str") or "").zfill(10)
	except Exception:
		pass
	return ""


def sec_get_cik_by_name(name: str) -> str:
	if not name:
		return ""
	data = _sec_company_tickers()
	try:
		for _, v in (data.items() if isinstance(data, dict) else []):
			title = v.get("title") or ""
			if name.lower() in title.lower():
				return str(v.get("cik_str") or "").zfill(10)
	except Exception:
		pass
	return ""


def sec_get_latest_filing_text(cik: str) -> str:
	if not cik:
		return ""
	subm_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
	logger.info("Fetching SEC submissions for CIK=%s", cik)
	subm = _sec_session().get(subm_url, headers=_sec_headers(), timeout=20).json()
	recent = ((subm.get("filings") or {}).get("recent") or {})
	forms = recent.get("form") or []
	accessions = recent.get("accessionNumber") or []
	primaries = recent.get("primaryDocument") or []
	preferred: List[int] = []
	for code in ["10-K", "20-F", "40-F", "10-Q", "6-K"]:
		preferred += [i for i, f in enumerate(forms) if f == code]
	for i in preferred:
		if i >= len(accessions) or i >= len(primaries):
			continue
		accession = (accessions[i] or "").replace("-", "")
		primary = (primaries[i] or "").strip()
		if not accession:
			continue
		candidates: List[str] = []
		if primary and primary.lower().endswith((".htm", ".html", ".txt")):
			candidates.append(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary}")
		# Fallback: probe the directory index.json for an html/txt document
		index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
		try:
			idx = _sec_session().get(index_url, headers=_sec_headers(), timeout=20)
			if idx.ok:
				data = idx.json()
				items = ((data.get("directory") or {}).get("item") or [])
				for it in items:
					href = (it.get("href") or "").strip()
					if href.lower().endswith((".htm", ".html")):
						candidates.append(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{href}")
				for it in items:
					href = (it.get("href") or "").strip()
					if href.lower().endswith((".txt",)):
						candidates.append(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{href}")
		except Exception:
			pass
		for url in candidates:
			try:
				logger.info("Fetching SEC filing %s", url)
				r = _sec_session().get(url, headers=_sec_headers(), timeout=30)
				r.raise_for_status()
				return r.text
			except Exception:
				logger.exception("Failed to fetch SEC filing %s", url)
				continue
	return ""


def _html_to_text(html: str) -> str:
	soup = BeautifulSoup(html or "", "lxml")
	for tag in soup(["script", "style", "footer", "nav", "table"]):
		tag.decompose()
	text = soup.get_text("\n", strip=True)
	text = re.sub(r"\n{2,}", "\n", text)
	text = re.sub(r"[ \t]+", " ", text)
	return text.strip()


def _extract_business_or_mda(text: str) -> str:
	if not text:
		return ""
	# Try Item 1. Business
	m = re.search(r"(?is)Item\s+1\.?\s*Business(.*?)(Item\s+1A\.?|Item\s+2\.)", text)
	if m:
		return (m.group(1) or "").strip()
	# Try MD&A from 10-Q structure
	m = re.search(r"(?is)Part\s+I\b.*?Item\s+2\.?\s*Management.*?Discussion.*?Analysis(.*?)(Item\s+3\.|Part\s+II\b)", text)
	if m:
		return (m.group(1) or "").strip()
	# Fallback: leading narrative chunk
	return text[:8000]


def sec_summarize_filing(html: str, openai_api_key: str) -> str:
	if not html:
		return ""
	text = _html_to_text(html)
	section = _extract_business_or_mda(text)
	if not section:
		return ""
	client = OpenAI(api_key=openai_api_key)
	system = (
		"Summarize ONLY the company's business (products/services, customers, end-markets, delivery model, geography). "
		"Ignore codes of ethics, insider trading policies, governance, compensation, exhibits, and legal boilerplate."
	)
	user = "Section text:\n" + section[:12000]
	try:
		resp = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
		)
		out = (resp.choices[0].message.content or "").strip()
		logger.info("SEC summary generated len=%s", len(out))
		return out
	except Exception:
		logger.exception("SEC summary failed, returning truncated section")
		return section[:1200]


def enrich_with_sec(
	target_name: str,
	comparables: List[Dict[str, Any]],
	openai_api_key: str,
	default_target_description: str,
) -> Dict[str, Any]:
	"""Fetch SEC filings for target and peers, summarize to enrich descriptions.

	Returns {"target_description": str, "comparables": List[comp with sec_summary]}.
	"""
	# Target
	target_cik = sec_get_cik_by_name(target_name)
	target_summary = ""
	if target_cik:
		try:
			filing_text = sec_get_latest_filing_text(target_cik)
			target_summary = sec_summarize_filing(filing_text, openai_api_key)
		except Exception:
			target_summary = ""
	enriched_target = target_summary or default_target_description
	# Peers
	enriched: List[Dict[str, Any]] = []
	sec_hits = 0
	for comp in comparables:
		sec_summary = ""
		ticker = (comp.get("ticker") or comp.get("symbol") or "").strip()
		cik = sec_get_cik_by_ticker(ticker) or sec_get_cik_by_name(comp.get("name") or "")
		if cik:
			try:
				filing_text = sec_get_latest_filing_text(cik)
				sec_summary = sec_summarize_filing(filing_text, openai_api_key)
				if sec_summary:
					sec_hits += 1
			except Exception:
				sec_summary = ""
		new_c = dict(comp)
		if sec_summary:
			new_c["sec_summary"] = sec_summary
		enriched.append(new_c)
	logger.info("SEC enrichment: peers=%s with_summary=%s", len(comparables), sec_hits)
	return {"target_description": enriched_target, "comparables": enriched}

def _cosine_similarity(a: List[float], b: List[float]) -> float:
	va = np.array(a)
	vb = np.array(b)
	den = (np.linalg.norm(va) * np.linalg.norm(vb))
	if den == 0:
		return 0.0
	return float(np.dot(va, vb) / den)


def _comp_similarity_text(comp: Dict[str, Any]) -> str:
	# Prefer SEC summary if available for richer text
	sec_summary = (comp.get("sec_summary") or "").strip()
	if sec_summary:
		return sec_summary
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
	try:
		embs = _get_embeddings_cached(texts, openai_api_key, model="text-embedding-3-small")
		target_vec = embs[0]
		updated: List[Dict[str, Any]] = []
		for idx, comp in enumerate(comparables, start=1):
			sim = _cosine_similarity(target_vec, embs[idx])
			# Map cosine [-1,1] -> [0,1]
			score01 = (sim + 1.0) / 2.0
			new_c = dict(comp)
			new_c["similarity_score"] = round(float(score01), 4)
			updated.append(new_c)
		if updated:
			vals = [c.get("similarity_score", 0.0) for c in updated if isinstance(c.get("similarity_score"), (int, float))]
			logger.info("Similarity scores computed: n=%s min=%.3f max=%.3f", len(updated), min(vals or [0.0]), max(vals or [0.0]))
		return {"comparables": updated}
	except Exception:
		# Fallback: keyword overlap using business_activity
		logger.exception("Embedding similarity failed; falling back to keyword overlap")
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


