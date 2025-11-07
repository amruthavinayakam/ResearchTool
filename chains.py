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
		model="gpt-4o",
		temperature=0,
		api_key=openai_key,
		model_kwargs={"response_format": {"type": "json_object"}},
	)
	classifier = ChatOpenAI(
		model="gpt-4o-mini",
		temperature=0,
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

	def _resolve_symbol(self, name: str, url: str) -> str:
		# Try search by company name first
		try:
			res = self._fh_get("/search", {"q": name})
			for item in (res.get("result") or []):
				if item.get("symbol"):
					logger.info("Resolved symbol by name: %s -> %s", name, item.get("symbol"))
					return item["symbol"]
		except Exception:
			pass
		# Fallback: try domain tokens
		domain = re.sub(r"^https?://", "", url).split("/")[0] if url else ""
		if domain:
			try:
				res = self._fh_get("/search", {"q": domain})
				for item in (res.get("result") or []):
					if item.get("symbol"):
						logger.info("Resolved symbol by domain: %s -> %s", domain, item.get("symbol"))
						return item["symbol"]
			except Exception:
				pass
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
				return {"name": name2, "url": url2, "context": industry, "title": name2, "symbol": sym}
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
				"Return ONLY PUBLICLY TRADED COMPANIES whose products/services and customer segments are similar to the target (NO PRIVATE COMPANIES). "
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
				"Return a JSON object with key 'comparables' containing 3 to 10 entries of PUBLICLY TRADED COMPANIES "
				"whose products/services and customer segments are similar to the target (NO PRIVATE COMPANIES). "
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
		cache_key = "classify:" + self._hash_obj({"target": {"name": target.get("name"), "desc": target.get("business_description")}, "cands": candidates})
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached
		logger.info("Classifying %s candidates for target=%s", len(candidates), target.get("name"))
		instruction = (
			"You are a strict classifier. From the provided candidate companies, select only those that are publicly traded "
			"and clearly relevant (similar products/services and customer segments) to the target company. "
			"Return JSON with key 'accepted' as a list of objects with keys: name, url, reason."
		)
		prompt = ChatPromptTemplate.from_messages(
			[
				("system", instruction),
				(
					"user",
					(
						"Target company summary:\n"
						"name: {target_name}\n"
						"description: {target_description}\n\n"
						"Candidates (name, url, snippet):\n{candidates_json}\n\n"
						"Return JSON with only the key 'accepted'. Example (keys quoted): "
						"{\"accepted\": [{\"name\": \"Company A\", \"url\": \"https://example.com\", \"reason\": \"why similar\"}]}"
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
					"candidates_json": json.dumps(candidates, ensure_ascii=False),
				}
			)
			text = resp.content if hasattr(resp, "content") else str(resp)
			data = json.loads((text or "{}").strip())
			accepted = data.get("accepted") or []
			logger.info("Classifier accepted=%s", len(accepted))
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
				base = {"name": name, "url": url}
				orig = orig_map.get((name.strip().lower() + "|" + url.strip().lower()))
				if orig and orig.get("symbol"):
					base["symbol"] = orig.get("symbol")
				if orig and orig.get("context"):
					base["context"] = orig.get("context")
				filtered.append(base)
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
		# Gather peers first (signals not required for Finnhub fetch)
		candidates = self._gather_candidates(
			name=target_name, url=target_url, desc=target_description, sic=target_sic, signals={}
		)
		# Take top N candidates (no similarity ranking)
		top_candidates = candidates[:TOP_CANDIDATES_FOR_LLM]
		logger.info("Gathered candidates total=%s top_for_llm=%s", len(candidates), len(top_candidates))

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
