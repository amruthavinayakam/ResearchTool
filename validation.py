from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from diskcache import Cache
from openai import OpenAI

from utils import clean_json_text

logger = logging.getLogger(__name__)

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

_UNKNOWN_MARKERS = {"", "unknown", "n/a", "na", "none", "null", "-", "tbd", "undisclosed"}
_PLACEHOLDER_URL_DOMAINS = {"sec.gov"}


def _sec_headers() -> Dict[str, str]:
	return {
		"User-Agent": "ComparableFinder/1.0 (contact: vamrutha.works@gmail.com)",
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
		"Connection": "keep-alive",
		"Referer": "https://www.sec.gov/",
	}


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
			link = tds[0].find("a", href=True) or tds[1].find("a", href=True)
			url = ""
			if link:
				href = link["href"]
				url = href if href.startswith("http") else ("https://www.sec.gov" + href)
			results.append({"name": company_text, "cik": cik_text, "url": url})
	if not results:
		for anchor in soup.find_all("a", href=True):
			href = anchor["href"]
			if "action=getcompany" not in href or "CIK=" not in href:
				continue
			name = (anchor.get_text(" ", strip=True) or "").strip()
			match = re.search(r"CIK=(\d+)", href)
			cik = match.group(1) if match else ""
			if not name or not cik:
				continue
			url = href if href.startswith("http") else ("https://www.sec.gov" + href)
			results.append({"name": name, "cik": cik, "url": url})
	return results


def _is_us_exchange(exchange: str) -> bool:
	ex = (exchange or "").upper()
	if not ex:
		return False
	return any(keyword in ex for keyword in _US_EXCHANGE_KEYWORDS)


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
			profile: Dict[str, str] = {}
		else:
			data = resp.json() or {}
			tickers = [str(t or "").strip() for t in (data.get("tickers") or []) if t]
			exchanges = [str(e or "").strip() for e in (data.get("exchanges") or []) if e]
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
		resp = requests.get(base, params=params, headers=_sec_headers(), timeout=20)
		resp.raise_for_status()
		page_results = _parse_sec_browse_page(resp.text or "")
		logger.info("SEC scrape page=%s results=%s", page, len(page_results))
		if not page_results:
			break
		all_results.extend(page_results)
		if len(page_results) < count:
			break
		page += 1
		offset += count

	seen = set()
	deduped: List[Dict[str, str]] = []
	for entry in all_results:
		cik = entry.get("cik") or ""
		if cik in seen:
			continue
		seen.add(cik)
		deduped.append(entry)

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
		entry
		for entry in deduped
		if (entry.get("ticker") or "").strip() and (entry.get("exchange") or "").strip()
	]
	if len(filtered) < len(deduped):
		logger.info("Removed %s companies missing ticker or exchange after SEC enrichment", len(deduped) - len(filtered))

	us_filtered = [entry for entry in filtered if _is_us_exchange(entry.get("exchange"))]
	if len(us_filtered) < len(filtered):
		logger.info("Removed %s companies not trading on US exchanges", len(filtered) - len(us_filtered))

	logger.info(
		"SEC scrape complete: SIC=%s total_parsed=%s unique=%s pages=%s",
		sic_code,
		len(all_results),
		len(us_filtered),
		page + 1,
	)
	for entry in us_filtered[:10]:
		logger.debug(
			"SEC company: name=%s cik=%s url=%s ticker=%s exchange=%s",
			entry.get("name"),
			entry.get("cik"),
			entry.get("url"),
			entry.get("ticker"),
			entry.get("exchange"),
		)
	return us_filtered


def _needs_llm_fill(value: Any) -> bool:
	if value is None:
		return True
	if isinstance(value, str):
		return value.strip().lower() in _UNKNOWN_MARKERS
	return False


def _needs_llm_url(value: Any) -> bool:
	if _needs_llm_fill(value):
		return True
	if not isinstance(value, str):
		return True
	val = value.strip()
	if not val:
		return True
	try:
		host = (urlparse(val).netloc or "").lower()
	except Exception:
		return True
	if not host:
		return True
	return any(host.endswith(domain) for domain in _PLACEHOLDER_URL_DOMAINS)


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
		"You are an equity research analyst. KEEP only U.S.-listed "
		"public peers/competitors of the target (NYSE, NASDAQ, AMEX) with clearly "
		"overlapping services, customer segments, and business model. "
		"DROP non-U.S. listings, private firms, ADRs/ETFs, shells/holdcos, "
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
				company for idx, company in enumerate(subset) if idx in keep_indices
			]
			if not filtered_subset:
				logger.info(
					"LLM filtering removed all %s candidates in batch starting at %s",
					len(subset),
					offset,
				)
			filtered_all.extend(filtered_subset)
		except Exception:
			logger.exception("filter_companies_with_llm failed for batch starting at %s", offset)
			filtered_all.extend(subset)

	logger.info("LLM filtering kept %s of %s candidates", len(filtered_all), len(companies))
	return filtered_all


def enrich_company_details_with_llm(
	target_name: str,
	target_description: str,
	comparables: List[Dict[str, Any]],
	openai_api_key: str,
	batch_size: int = 6,
) -> List[Dict[str, Any]]:
	if not comparables or not openai_api_key:
		return comparables

	client = OpenAI(api_key=openai_api_key)
	target_indices = [
		idx
		for idx, comp in enumerate(comparables)
		if _needs_llm_url(comp.get("url"))
		or _needs_llm_fill(comp.get("business_activity"))
		or _needs_llm_fill(comp.get("customer_segment"))
	]
	if not target_indices:
		return comparables

	logger.info("Enriching %s comparables via LLM", len(target_indices))
	updated = [dict(comp) for comp in comparables]

	for start in range(0, len(target_indices), batch_size):
		chunk_indices = target_indices[start : start + batch_size]
		payload = []
		for idx in chunk_indices:
			comp = comparables[idx]
			payload.append(
				{
					"index": idx,
					"name": comp.get("name") or "",
					"ticker": comp.get("ticker") or "",
					"exchange": comp.get("exchange") or "",
					"current_url": comp.get("url") or "",
					"business_activity": comp.get("business_activity") or "",
					"customer_segment": comp.get("customer_segment") or "",
					"SIC_industry": comp.get("SIC_industry") or "",
				}
			)
		user_prompt = (
			"Fill in missing details for comparable public companies. "
			"For each record, replace only fields that are empty or placeholders (unknown, n/a, none, tbd, null, '-'). "
			"Provide the official corporate website (replace SEC or filing links with the company's primary domain). "
			"Keep business_activity concise (<=35 words) describing main offerings. "
			"Summarize customer_segment in <=20 words focusing on primary buyers/end-users. "
			"If the official site cannot be found confidently, set url to 'unknown'. "
			"Respond with strict JSON: {\"enriched\": [{\"index\": int, \"url\": str, \"business_activity\": str, \"customer_segment\": str}, ...]}. "
			"No extra commentary.\n\n"
			"Target company context:\n"
			f"name: {target_name}\n"
			f"description: {target_description[:500]}\n\n"
			f"Comparables needing enrichment:\n{json.dumps(payload, ensure_ascii=False)}"
		)
		try:
			resp = client.chat.completions.create(
				model="gpt-4o",
				messages=[
					{
						"role": "system",
						"content": (
							"You are a meticulous equity research associate. "
							"Fill missing attributes using reliable public knowledge. "
							"Return JSON only."
						),
					},
					{"role": "user", "content": user_prompt},
				],
				max_tokens=800,
			)
			text = resp.choices[0].message.content if resp.choices else "{}"
			data = json.loads(clean_json_text(text))
			results = data.get("enriched") if isinstance(data, dict) else []
		except Exception:
			logger.exception("LLM enrichment failed for batch starting at index %s", chunk_indices[0])
			continue
		if not isinstance(results, list):
			continue
		for item in results:
			if not isinstance(item, dict):
				continue
			try:
				idx = int(item.get("index"))
			except Exception:
				continue
			if idx not in chunk_indices:
				continue
			entry = updated[idx]
			url_val = (item.get("url") or "").strip()
			if url_val and _needs_llm_url(entry.get("url")):
				entry["url"] = url_val
			activity_val = (item.get("business_activity") or "").strip()
			if activity_val and _needs_llm_fill(entry.get("business_activity")):
				entry["business_activity"] = activity_val
			segment_val = (item.get("customer_segment") or "").strip()
			if segment_val and _needs_llm_fill(entry.get("customer_segment")):
				entry["customer_segment"] = segment_val
	return updated


def map_sic_with_llm(user_input: str, openai_api_key: str) -> Dict[str, Any]:
	if not openai_api_key:
		raise ValueError("OpenAI API key is required for SIC mapping.")

	client = OpenAI(api_key=openai_api_key)
	instruction = (
		"You are an expert on SEC Standard Industrial Classification (SIC) codes. "
		"Use at most two SIC codes for the target. If the user supplies SICs, they may provide one or two codes separated by a comma; "
		"treat the first as primary and the second as optional. The model must return no more than two SIC codes (strictly 4-digit numeric); "
		"if uncertain, return only one. Respond STRICTLY in JSON with a key 'matches' containing an array (max length 2). "
		"Each item must include sic_code (string, 4 digits), sic_name (official description), and confidence (float between 0 and 1)."
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

	if isinstance(data, list):
		matches_raw = data
	elif isinstance(data, dict):
		if isinstance(data.get("matches"), list):
			matches_raw = data["matches"]
		elif data.get("sic_code"):
			matches_raw = [data]
		else:
			matches_raw = []
	else:
		matches_raw = []

	if not isinstance(matches_raw, list) or not matches_raw:
		logger.error("map_sic_with_llm: response missing valid matches structure: %s", data)
		raise ValueError("LLM response missing 'matches' array.")

	cleaned: List[Dict[str, Any]] = []
	for item in matches_raw:
		if not isinstance(item, dict):
			continue
		code = str(item.get("sic_code") or "").strip()
		name = (item.get("sic_name") or "").strip()
		confidence = item.get("confidence")
		if not re.fullmatch(r"\d{4}", code):
			logger.warning("map_sic_with_llm: skipping invalid sic_code '%s'", code)
			continue
		cleaned.append(
			{
				"sic_code": code,
				"sic_name": name or "-",
				"confidence": confidence,
			}
		)
		if len(cleaned) == 2:
			break

	if not cleaned:
		logger.error("map_sic_with_llm: no valid SIC codes returned for input '%s'", user_input)
		raise ValueError(f"LLM did not return any valid 4-digit SIC codes for input '{user_input}'.")

	logger.info("map_sic_with_llm: LLM mapped '%s' -> %s", user_input, cleaned)
	return {"matches": cleaned}


def fetch_serp_peers(
	target_name: str,
	serp_api_key: str,
	max_results: int = 10,
) -> List[Dict[str, str]]:
	name = (target_name or "").strip()
	if not name or not serp_api_key:
		return []

	queries = [
		f"{name} publicly traded competitors",
		f"{name} publicly traded peers",
	]
	results: List[Dict[str, str]] = []
	seen_links: set[str] = set()
	for query in queries:
		params = {
			"engine": "google",
			"q": query,
			"api_key": serp_api_key,
			"num": max_results,
		}
		try:
			resp = requests.get("https://serpapi.com/search", params=params, timeout=25)
			resp.raise_for_status()
			payload = resp.json()
		except Exception:
			logger.exception("SerpAPI request failed for query '%s'", query)
			continue
		organic = payload.get("organic_results")
		if not isinstance(organic, list):
			continue
		for item in organic:
			if not isinstance(item, dict):
				continue
			link = (item.get("link") or "").strip()
			title = (item.get("title") or "").strip()
			if not link or link in seen_links:
				continue
			seen_links.add(link)
			display = title
			if " - " in title:
				display = title.split(" - ", 1)[0].strip()
			elif "|" in title:
				display = title.split("|", 1)[0].strip()
			if not display:
				display = title or link
			results.append(
				{
					"name": display,
					"url": link,
					"query": query,
					"title": title,
					"snippet": item.get("snippet") or "",
				}
			)
	logger.info("SerpAPI peers fetched %s results for target '%s'", len(results), name)
	return results

