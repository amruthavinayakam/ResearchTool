from __future__ import annotations

import json
import math
import re
import concurrent.futures
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import hashlib
import requests_cache
from diskcache import Cache
import logging

from utils import (
	retry_with_exponential_backoff,
	clean_json_text,
	unique_by_key,
)


MAX_CANDIDATES = 60
TOP_CANDIDATES_FOR_LLM = 40
CACHE_EXPIRE_SECONDS = 86400

# Install transparent HTTP cache (covers requests + Session)
requests_cache.install_cache("finnhub_cache", expire_after=CACHE_EXPIRE_SECONDS)
logging.getLogger(__name__).info("requests-cache enabled (finnhub_cache, ttl=%s)", CACHE_EXPIRE_SECONDS)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_llm_disk_cache() -> Cache:
	cache = Cache("llm_cache")
	logger.info("LLM disk cache initialized at llm_cache")
	return cache


def _build_queries(name: str, url: str, desc: str, sic: str, signals: Dict[str, Any]) -> List[str]:
	q: List[str] = []
	keywords = [k for k in (signals.get("keywords") or [])][:5]
	industries = [k for k in (signals.get("industry_groups") or [])][:4]
	segments = [k for k in (signals.get("customer_segments") or [])][:3]
	products = [k for k in (signals.get("products") or [])][:3]
	business_focus = (signals.get("business_focus") or "").strip()
	# Generic
	q.append(f"competitors of {name}")
	q.append(f"{name} similar companies public")

	# Domain-aware patterns from extracted signals (business_focus + industry groups + keywords)
	domain_terms = ([business_focus] if business_focus else []) + industries + keywords
	if domain_terms:
		main_domain = ", ".join(domain_terms[:3])
		q.append(f"public companies operating in {main_domain}")
		q.append(f"competitors in {main_domain} industry similar to {name}")
		q.append(f"companies with similar products or customer segments as {name}")
	# Industry-driven
	for ind in industries:
		q.append(f"top public companies in {ind}")
		q.append(f"public companies {ind} sector peers")
	# Keyword/product-driven
	for kw in keywords[:3]:
		q.append(f"public companies {kw} competitors")
	for prod in products:
		q.append(f"public companies offering {prod}")
	# Segment-driven
	for seg in segments:
		q.append(f"public companies serving {seg}")
	# URL-derived
	if url:
		domain = re.sub(r"^https?://", "", url).split("/")[0]
		q.append(f"site:linkedin.com {domain} competitors")
	# Description fallback
	if desc:
		q.append(desc[:120] + " comparable public companies")
	# Deduplicate while preserving order
	seen = set()
	result: List[str] = []
	for item in q:
		k = item.lower()
		if k in seen:
			continue
		seen.add(k)
		result.append(item)
	return result


@st.cache_resource
def get_llm_clients(openai_key: str) -> tuple[ChatOpenAI, ChatOpenAI]:
	# Use JSON response format to improve structured outputs
	llm = ChatOpenAI(
		model="gpt-5",
		temperature=0,
		max_tokens=900,
		api_key=openai_key,
		model_kwargs={"response_format": {"type": "json_object"}},
	)
	classifier = ChatOpenAI(
		model="gpt-5",
		temperature=0,
		max_tokens=400,
		api_key=openai_key,
		model_kwargs={"response_format": {"type": "json_object"}},
	)
	return llm, classifier


class ComparableFinder:
	def __init__(self, openai_api_key: str, serpapi_api_key: str | None = None, finnhub_api_key: str | None = None) -> None:
		self.openai_api_key = openai_api_key
		self.serpapi_api_key = serpapi_api_key or ""
		self.finnhub_api_key = finnhub_api_key or ""
		# HTTP session with connection pooling and basic retries
		self.http = requests.Session()
		retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
		adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
		self.http.mount("https://", adapter)
		self.http.mount("http://", adapter)
		# Disk cache for LLM outputs
		self.llm_cache = get_llm_disk_cache()
		# Cached LLM clients
		llm, classifier = get_llm_clients(openai_api_key)
		self.llm = llm
		self.classifier_llm = classifier
		logger.info("ComparableFinder init: finnhub_key=%s, openai_key=%s", bool(self.finnhub_api_key), bool(self.openai_api_key))

	def _hash_obj(self, obj: Any) -> str:
		try:
			data = json.dumps(obj, ensure_ascii=False, sort_keys=True)
		except Exception:
			data = str(obj)
		return hashlib.sha256(data.encode("utf-8")).hexdigest()

	def _ensure_valid_output(self, structured: Dict[str, Any], fallback_candidates: List[Dict[str, str]]) -> Dict[str, Any]:
		"""Ensure LLM output has required keys; fallback to candidates if empty/invalid."""
		required = ["name", "url", "exchange", "ticker", "business_activity", "customer_segment", "SIC_industry"]
		result: List[Dict[str, Any]] = []
		try:
			comps = structured.get("comparables")
		except Exception:
			comps = None
		if isinstance(comps, list):
			for item in comps:
				if not isinstance(item, dict):
					continue
				name = (item.get("name") or "").strip()
				url = (item.get("url") or "").strip() or "unknown"
				if not name:
					continue
				clean: Dict[str, Any] = {
					"name": name,
					"url": url or "unknown",
					"exchange": (item.get("exchange") or "unknown").strip() or "unknown",
					"ticker": (item.get("ticker") or "unknown").strip() or "unknown",
					"business_activity": (item.get("business_activity") or "unknown").strip() or "unknown",
					"customer_segment": (item.get("customer_segment") or "unknown").strip() or "unknown",
					"SIC_industry": (item.get("SIC_industry") or "unknown").strip() or "unknown",
				}
				# Keep only if all required keys present and non-empty
				if all((isinstance(clean[k], str) and clean[k].strip()) for k in required):
					result.append(clean)
		logger.info("LLM structured count=%s valid=%s", (len(comps) if isinstance(comps, list) else 0), len(result))
		# If insufficient, synthesize from fallback candidates
		if len(result) < 3 and isinstance(fallback_candidates, list):
			logger.warning("Insufficient structured entries (%s) — synthesizing from %s fallback candidates", len(result), len(fallback_candidates))
			for c in fallback_candidates:
				name = (c.get("name") or "").strip()
				url = (c.get("url") or "").strip() or "unknown"
				if not name:
					continue
				context = (c.get("context") or "").strip()
				result.append(
					{
						"name": name,
						"url": url or "unknown",
						"exchange": "unknown",
						"ticker": (c.get("symbol") or "").strip() or "unknown",
						"business_activity": "unknown",
						"customer_segment": "unknown",
						"SIC_industry": context or "unknown",
					}
				)
				if len(result) >= 5:
					break
		# Bound to 3..10 per UI contract
		if len(result) > 10:
			result = result[:10]
		logger.info("Normalized comparables count=%s", len(result))
		return {"comparables": result}

	def _cache_get(self, key: str) -> Any:
		try:
			val = self.llm_cache.get(key, default=None)
			if val is not None:
				logger.info("Disk cache HIT for key=%s", key.split(":", 1)[0])
			else:
				logger.info("Disk cache MISS for key=%s", key.split(":", 1)[0])
			return val
		except Exception:
			return None

	def _cache_set(self, key: str, value: Any, expire: int = CACHE_EXPIRE_SECONDS) -> None:
		try:
			self.llm_cache.set(key, value, expire=expire)
		except Exception:
			pass

	def _fh_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any] | List[Any]:
		if not self.finnhub_api_key:
			raise ValueError("Finnhub API key is required")
		url = f"https://finnhub.io/api/v1{path}"
		try_params = dict(params)
		try_params.pop("token", None)
		logger.info("Finnhub GET %s params=%s", path, list(try_params.keys()))
		resp = self.http.get(url, params={**params, "token": self.finnhub_api_key}, timeout=12)
		resp.raise_for_status()
		return resp.json() or {}

	def _extract_signals(self, description: str) -> Dict[str, Any]:
		"""Extract concise signals (keywords, industry groups, products, segments, business_focus) using LLM JSON."""
		cache_key = "signals:" + self._hash_obj({"text": description})
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached
		logger.info("Extracting signals for desc_len=%s", len(description or ""))
		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					(
						"Extract business signals from the description. Return JSON with keys: "
						"keywords (5-12), industry_groups (1-5 names, not codes), products (1-6), customer_segments (1-6), "
						"business_focus (3-5 words summarizing the domain/industry)."
					),
				),
				("user", "Description: {text}"),
			]
		)
		chain = prompt | self.llm
		try:
			resp = chain.invoke({"text": description})
			text = resp.content if hasattr(resp, "content") else str(resp)
			data = json.loads((text or "{}").strip())
			result = {
				"keywords": data.get("keywords") or [],
				"industry_groups": data.get("industry_groups") or [],
				"products": data.get("products") or [],
				"customer_segments": data.get("customer_segments") or [],
				"business_focus": (data.get("business_focus") or "").strip(),
			}
			self._cache_set(cache_key, result)
			logger.info("Signals extracted: keywords=%s industry_groups=%s products=%s segments=%s",
			            len(result["keywords"]), len(result["industry_groups"]), len(result["products"]), len(result["customer_segments"]))
			return result
		except Exception:
			logger.exception("Signal extraction failed")
			return {"keywords": [], "industry_groups": [], "products": [], "customer_segments": [], "business_focus": ""}

	def _gpt_search_candidates(self, name: str, sic_name: str, description: str, url: str) -> List[Dict[str, str]]:
		"""Use GPT-5 to propose peer/competitor candidates in the same SIC industry group.

		Returns lightweight candidates: [{name, url, context, symbol?}].
		"""
		cache_key = "gpt_search:" + self._hash_obj({"name": name, "sic": sic_name, "desc": description, "url": url})
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached
		logger.info("GPT search for peers: name=%s sic=%s", name, sic_name)
		instruction = (
			"You are a strict research assistant. Find ONLY U.S. publicly traded competitors and peers of the target in the SAME SIC industry "
			"(use the provided SIC name/code; if code, infer the group name). "
			"Return 3-10 JSON objects with keys: name, url, context, and optional symbol (ticker). Exchanges allowed: NYSE, NASDAQ, AMEX. "
			"Exclude private, non-U.S., ADRs, ETFs/funds, suppliers, and customers. Avoid duplicates."
		)
		user = (
			"Target:\n"
			f"name: {name}\n"
			f"url: {url}\n"
			f"sic: {sic_name}\n"
			f"description: {description}\n\n"
			"Return JSON like {{\"candidates\": [{{\"name\":\"...\",\"url\":\"https://...\",\"context\":\"short reason\",\"symbol\":\"TICK\"}}]}}"
		)
		prompt = ChatPromptTemplate.from_messages([("system", instruction), ("user", user)])
		chain = prompt | self.llm
		try:
			resp = chain.invoke({})
			text = resp.content if hasattr(resp, "content") else str(resp)
			data = json.loads((text or "{}").strip())
			items = data.get("candidates") or []
			result: List[Dict[str, str]] = []
			for it in items:
				if not isinstance(it, dict):
					continue
				nm = (it.get("name") or "").strip()
				url = (it.get("url") or "").strip()
				if not nm or not url:
					continue
				entry: Dict[str, str] = {"name": nm, "url": url, "context": (it.get("context") or "").strip()}
				sym = (it.get("symbol") or "").strip()
				if sym:
					entry["symbol"] = sym
				result.append(entry)
			# bound size
			if len(result) > MAX_CANDIDATES:
				result = result[:MAX_CANDIDATES]
			# cache
			self._cache_set(cache_key, result)
			logger.info("GPT search candidates=%s", len(result))
			return result
		except Exception:
			logger.exception("GPT-5 SIC-based search failed")
			return []

	def _resolve_symbol(self, name: str, url: str) -> str:
		logger.info("Resolve symbol start: name=%s url=%s", name, url)
		# Try search by company name first
		try:
			res = self._fh_get("/search", {"q": name})
			results = (res.get("result") or []) if isinstance(res, dict) else []
			logger.info("Finnhub search by name: query=%s candidates=%s", name, len(results))
			if results:
				preview = []
				for item in results[:5]:
					sym = (item.get("symbol") or "").strip()
					desc = (item.get("description") or item.get("displaySymbol") or "").strip()
					if sym or desc:
						preview.append(f"{sym}|{desc}")
				if preview:
					logger.info("Name search preview (top5): %s", "; ".join(preview)[:500])
			for idx, item in enumerate(results):
				if item.get("symbol"):
					logger.info("Resolved symbol by name at idx=%s: %s -> %s", idx, name, item.get("symbol"))
					return item["symbol"]
			logger.info("No symbol found in search-by-name results for %s", name)
		except Exception:
			logger.exception("Finnhub search by name failed for %s", name)
		# Fallback: try domain tokens
		domain = re.sub(r"^https?://", "", url).split("/")[0] if url else ""
		if domain:
			try:
				res = self._fh_get("/search", {"q": domain})
				results = (res.get("result") or []) if isinstance(res, dict) else []
				logger.info("Finnhub search by domain: query=%s candidates=%s", domain, len(results))
				if results:
					preview = []
					for item in results[:5]:
						sym = (item.get("symbol") or "").strip()
						desc = (item.get("description") or item.get("displaySymbol") or "").strip()
						if sym or desc:
							preview.append(f"{sym}|{desc}")
					if preview:
						logger.info("Domain search preview (top5): %s", "; ".join(preview)[:500])
				for idx, item in enumerate(results):
					if item.get("symbol"):
						logger.info("Resolved symbol by domain at idx=%s: %s -> %s", idx, domain, item.get("symbol"))
						return item["symbol"]
				logger.info("No symbol found in search-by-domain results for %s", domain)
			except Exception:
				logger.exception("Finnhub search by domain failed for %s", domain)
		logger.warning("Failed to resolve symbol for %s (%s)", name, url)
		return ""

	def _gather_candidates(self, name: str, url: str, desc: str, sic: str, signals: Dict[str, Any]) -> List[Dict[str, str]]:
		# Resolve target symbol, then fetch peers from Finnhub
		symbol = self._resolve_symbol(name, url)
		if not symbol:
			return []
		try:
			peers = self._fh_get("/stock/peers", {"symbol": symbol})
		except Exception:
			peers = []
		if not isinstance(peers, list):
			peers = []
		# Fetch peer profiles in parallel with bounded concurrency
		def fetch_profile(sym: str) -> Dict[str, str] | None:
			try:
				profile = self._fh_get("/stock/profile2", {"symbol": sym})
				name2 = (profile.get("name") or "").strip()
				if not name2:
					return None
				url2 = (profile.get("weburl") or "").strip()
				industry = (profile.get("finnhubIndustry") or "").strip()
				country = (profile.get("country") or "").strip()
				# Enforce US-only at the source
				if country and country.upper() != "US":
					return None
				return {"name": name2, "url": url2, "context": industry, "title": name2, "symbol": sym, "country": country}
			except Exception:
				return None
		candidates: List[Dict[str, str]] = []
		syms = list(peers[:MAX_CANDIDATES])
		logger.info("Peers fetched: %s (capped=%s)", len(peers) if isinstance(peers, list) else 0, len(syms))
		if syms:
			with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
				for res in ex.map(fetch_profile, syms, chunksize=4):
					if res:
						candidates.append(res)
		# Deduplicate by name
		candidates = unique_by_key(candidates, key_fn=lambda x: re.sub(r"[^A-Za-z0-9]", "", x["name"]).lower())
		logger.info("Candidate profiles fetched=%s deduped=%s", len(syms), len(candidates))
		return candidates[:MAX_CANDIDATES]

	def _compose_prompt(self, target: Dict[str, str], candidates: List[Dict[str, str]]) -> ChatPromptTemplate:
		if candidates:
			instruction = (
				"You are given a target company and a list of candidate companies. "
				"Return a JSON object with key 'comparables' containing 3 to 10 entries. "
				"Choose ONLY from the provided candidates; do not add companies not present in the candidate list. "
				"Return ONLY PUBLICLY TRADED U.S. COMPANIES (listed on NYSE, NASDAQ, or AMEX) whose products/services and customer segments are similar to the target. NO PRIVATE OR NON-U.S. COMPANIES. "
				"Use provided snippets and any provided ticker/symbol fields to infer fields. If ticker/exchange cannot be determined confidently, write 'unknown'. "
				"Fields per comparable (exact keys): name, url, exchange, ticker, business_activity, customer_segment, SIC_industry. "
				"For SIC_industry, output the SIC industry group name(s), absolutely do NOT include any numeric SIC codes or numbers, "
				"comma-separated when multiple apply."
			)
			user_msg = (
				"Target company:\n"
				"name: {target_name}\n"
				"url: {target_url}\n"
				"business_description: {target_description}\n"
				"SIC_industry: {target_sic}\n\n"
				"Candidates (include symbol when available):\n{candidates_json}\n\n"
				"Return JSON with a single key 'comparables' -> list[3..10] of objects with keys: name, url, exchange, ticker, business_activity, customer_segment, SIC_industry."
			)
		else:
			instruction = (
				"You are given a target company. "
				"Return a JSON object with key 'comparables' containing 3 to 10 entries of PUBLICLY TRADED U.S. COMPANIES (NYSE, NASDAQ, AMEX only). "
				"Companies must be U.S. registrants (SIC is U.S.-specific). NO PRIVATE OR NON-U.S. COMPANIES. "
				"If ticker/exchange cannot be determined confidently, write 'unknown'. "
				"Fields per comparable (exact keys): name, url, exchange, ticker, business_activity, customer_segment, SIC_industry. "
				"For SIC_industry, output the SIC industry group name(s), absolutely do NOT include any numeric SIC codes or numbers."
			)
			user_msg = (
				"Target company:\n"
				"name: {target_name}\n"
				"url: {target_url}\n"
				"business_description: {target_description}\n"
				"SIC_industry: {target_sic}\n\n"
				"No candidates were available. Generate comparable public companies directly from the target description.\n"
				"Return JSON with a single key 'comparables' -> list[3..10] with fields: name, url, exchange, ticker, business_activity, customer_segment, SIC_industry."
			)
		prompt = ChatPromptTemplate.from_messages(
			[
				("system", instruction),
				("user", user_msg),
			]
		)
		return prompt

	def _classify_candidates(self, target: Dict[str, str], candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
		"""Filter to relevant, publicly traded candidates using a cheaper classifier model."""
		# Guard: nothing to classify
		if not candidates:
			return []
		# Extract signals (cached) to build stronger guardrails
		signals = self._extract_signals(target.get("business_description") or "")
		cache_key = "classify:" + self._hash_obj({"target": {"name": target.get("name"), "desc": target.get("business_description"), "url": target.get("url"), "sic": target.get("sic")}, "cands": candidates})
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached
		logger.info("Classifying %s candidates for target=%s", len(candidates), target.get("name"))
		instruction = (
			"You are a strict classifier. Select ONLY candidates that are publicly traded U.S. companies (NYSE, NASDAQ, AMEX) and are direct peers/competitors "
			"with similar products/services and customer segments, operating in the SAME SIC industry group as the target. "
			"Leverage the target's name, URL domain and brief description to judge fit. "
			"REJECT: private, non-U.S., ADRs, ETFs/funds, holding companies with no operations, suppliers, distributors, customers, consultants to the target, "
			"and entities whose primary business model is orthogonal to the target. "
			"If confidence is low, do not accept. "
			"Return STRICT JSON with key 'accepted' as a list of 3-12 objects, each with keys: name, url, reason."
		)
		prompt = ChatPromptTemplate.from_messages(
			[
				("system", instruction),
				(
					"user",
					(
						"Target company summary:\n"
						"name: {target_name}\n"
						"url: {target_url}\n"
						"description: {target_description}\n"
						"sic: {target_sic}\n\n"
						"Candidates (name, url, snippet):\n{candidates_json}\n\n"
						"Return JSON with only the key 'accepted'. Example (keys quoted): "
						"{{\"accepted\": [{{\"name\": \"Company A\", \"url\": \"https://example.com\", \"reason\": \"why similar\"}}]}}"
					),
				),
			]
		)
		chain = prompt | self.classifier_llm
		try:
			resp = chain.invoke(
				{
					"target_name": target["name"],
					"target_description": target["business_description"],
					"target_sic": target.get("sic", ""),
					"target_url": target.get("url", ""),
					"candidates_json": json.dumps(candidates, ensure_ascii=False),
				}
			)
			text = resp.content if hasattr(resp, "content") else str(resp)
			data = json.loads((text or "{}").strip())
			accepted = data.get("accepted") or []
			logger.info("Classifier accepted=%s", len(accepted))
			# If too few accepted, escalate to stronger model for accuracy
			if len(accepted) < 3:
				logger.info("Classifier escalation: using main LLM due to low accepted count")
				chain2 = prompt | self.llm
				resp2 = chain2.invoke(
					{
						"target_name": target["name"],
						"target_description": target["business_description"],
						"target_sic": target.get("sic", ""),
						"target_url": target.get("url", ""),
						"candidates_json": json.dumps(candidates, ensure_ascii=False),
					}
				)
				text2 = resp2.content if hasattr(resp2, "content") else str(resp2)
				data2 = json.loads((text2 or "{}").strip())
				accepted = data2.get("accepted") or accepted
				logger.info("Classifier escalation accepted=%s", len(accepted))
			# Build lookup to preserve symbol/context
			def _keyify(x: Dict[str, str]) -> str:
				return ((x.get("name") or "").strip().lower() + "|" + (x.get("url") or "").strip().lower())
			orig_map = {_keyify(c): c for c in candidates}
			# Map to simple candidate structures for the formatter stage, preserving symbol if available
			filtered: List[Dict[str, str]] = []
			for item in accepted:
				name = (item.get("name") or "").strip()
				url = (item.get("url") or "").strip()
				if not (name and url):
					continue
				# drop self or near-duplicates of target
				if name.lower() == (target.get("name") or "").strip().lower():
					continue
				base = {"name": name, "url": url}
				orig = orig_map.get((name.strip().lower() + "|" + url.strip().lower()))
				if orig and orig.get("symbol"):
					base["symbol"] = orig.get("symbol")
				if orig and orig.get("context"):
					base["context"] = orig.get("context")
				filtered.append(base)
			# Programmatic guardrail: drop clearly irrelevant by token overlap with target name/description/SIC and domains
			def _tokset(s: str) -> set[str]:
				return {t.lower() for t in (s or "").split() if t.isalpha() or t.isalnum()}
			def _domain_tokens(u: str) -> set[str]:
				try:
					from urllib.parse import urlparse as _p
					host = _p(u or "").netloc or ""
					parts = [p for p in host.split(".") if p]
					stop = {"www", "com", "net", "org", "io", "inc", "corp", "co", "ltd"}
					return {p.lower() for p in parts if p.lower() not in stop}
				except Exception:
					return set()
			target_tokens = _tokset((target.get("business_description") or "") + " " + (target.get("sic") or ""))
			target_tokens |= _tokset(target.get("name") or "")
			target_tokens |= _domain_tokens(target.get("url") or "")
			# include signal-derived keywords/industry terms
			try:
				signal_terms = []
				for lst in [signals.get("keywords") or [], signals.get("industry_groups") or []]:
					for item in lst:
						signal_terms += [w for w in str(item).split() if w]
				target_tokens |= {w.lower() for w in signal_terms}
			except Exception:
				pass
			if filtered and target_tokens:
				refined: List[Dict[str, str]] = []
				for cand in filtered:
					orig = orig_map.get(_keyify(cand)) or {}
					ctx = (orig.get("context") or "")
					cand_tokens = _tokset(cand.get("name") or "") | _tokset(ctx) | _domain_tokens(cand.get("url") or "")
					overlap = len(target_tokens & cand_tokens)
					if overlap >= 4:
						refined.append(cand)
				# only apply if we still have a reasonable set
				if len(refined) >= 3:
					filtered = refined
			# Bound size
			if len(filtered) > 15:
				filtered = filtered[:15]
			self._cache_set(cache_key, filtered)
			return filtered
		except Exception:
			logger.exception("Classifier failed; returning head candidates")
			return candidates[:10]

	@retry_with_exponential_backoff(max_retries=2)
	def _call_llm(self, prompt: ChatPromptTemplate, variables: Dict[str, Any]) -> Dict[str, Any]:
		cache_key = "format:" + self._hash_obj({"vars": variables})
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached
		chain = prompt | self.llm
		logger.info("Calling formatter LLM with %s candidates", len(json.loads(variables["candidates_json"])) if "candidates_json" in variables else "n/a")
		resp = chain.invoke(variables)
		text = resp.content if hasattr(resp, "content") else str(resp)
		data = json.loads(clean_json_text(text))
		self._cache_set(cache_key, data)
		return data

	def find_comparables(
		self,
		target_name: str,
		target_url: str,
		target_description: str,
		target_sic: str = "",
	) -> Dict[str, Any]:
		# Gather peers from Finnhub and GPT-5 in parallel to reduce latency
		with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
			fh_future = ex.submit(self._gather_candidates, target_name, target_url, target_description, target_sic, {})
			gpt_future = ex.submit(self._gpt_search_candidates, target_name, target_sic, target_description, target_url)
			fh_candidates = fh_future.result() or []
			gpt_candidates = gpt_future.result() or []
		# Combine and dedupe by normalized name
		combined = (fh_candidates or []) + (gpt_candidates or [])
		combined = unique_by_key(combined, key_fn=lambda x: re.sub(r"[^A-Za-z0-9]", "", (x.get("name") or "")).lower())
		# Cap for downstream LLM
		top_candidates = combined[:TOP_CANDIDATES_FOR_LLM]
		logger.info("Candidates: finnhub=%s gpt=%s combined=%s top_for_llm=%s",
		            len(fh_candidates or []), len(gpt_candidates or []), len(combined), len(top_candidates))

		# Tier 1: run classifier and signals in parallel to overlap LLM latencies
		target = {
			"name": target_name,
			"url": target_url,
			"business_description": target_description,
			"sic": target_sic,
		}
		with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
			signals_future = ex.submit(self._extract_signals, target_description)
			classify_future = ex.submit(self._classify_candidates, target, top_candidates)
			filtered_candidates = classify_future.result() or top_candidates
			logger.info("Filtered candidates count=%s", len(filtered_candidates))
			# Tier 2: format/enrich using stronger model (can overlap with signals extraction)
			prompt = self._compose_prompt(target, filtered_candidates)
			variables = {
				"target_name": target_name,
				"target_url": target_url,
				"target_description": target_description,
				"target_sic": target_sic,
				"candidates_json": json.dumps(filtered_candidates, ensure_ascii=False),
			}
			format_future = ex.submit(self._call_llm, prompt, variables)
			_ = signals_future.result()  # ensure any signal extraction completes (cache warmed)
			structured = format_future.result()
			return self._ensure_valid_output(structured, filtered_candidates or top_candidates)
