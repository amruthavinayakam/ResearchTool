import json
import os
from typing import List, Dict, Any

import streamlit as st
import time
import logging

# Configure logging once
if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

from chains import ComparableFinder, get_llm_clients
from validation import (
	validate_and_normalize_output,
	ground_ticker_exchange,
	add_reasoning_notes,
 	compute_similarity_scores,
 	enrich_similarity_scores,
 	normalize_similarity_scores,
	enrich_with_sec,
	filter_us_sec_registrants,
	filter_us_active_listed,
)


st.set_page_config(page_title="Comparable Companies Finder", page_icon="📈", layout="wide")


def init_sidebar_keys() -> Dict[str, str]:
	with st.sidebar:
		st.header("API Keys")
		# Prefer environment variables if available
		env_openai = os.getenv("OPENAI_API_KEY") or ""
		env_finnhub = os.getenv("FINNHUB_API_KEY") or ""
		env_av = os.getenv("ALPHAVANTAGE_API_KEY") or ""
		if env_openai:
			openai_api_key = env_openai
			st.caption("Using OPENAI_API_KEY from environment.")
		else:
			openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
		if env_finnhub:
			finnhub_api_key = env_finnhub
			st.caption("Using FINNHUB_API_KEY from environment.")
		else:
			finnhub_api_key = st.text_input("Finnhub API Key", type="password", key="finnhub_api_key")
		if env_av:
			alphavantage_api_key = env_av
			st.caption("Using ALPHAVANTAGE_API_KEY from environment.")
		else:
			alphavantage_api_key = st.text_input("Alpha Vantage API Key", type="password", key="alphavantage_api_key")
		st.caption("Keys are kept only in this session state.")
		return {"openai_api_key": openai_api_key, "finnhub_api_key": finnhub_api_key, "alphavantage_api_key": alphavantage_api_key}


def main() -> None:
	st.title("Comparable Public Companies Generator")
	st.write(
		"Provide target company details. The app will use SerpAPI and OpenAI to propose comparable publicly traded companies."
	)

	keys = init_sidebar_keys()

	# Warm up cached resources so subsequent reruns are instant
	try:
		_ = get_llm_clients(keys["openai_api_key"]) if keys.get("openai_api_key") else None
	except Exception:
		pass

	with st.form("company_form"):
		col1, col2 = st.columns(2)
		with col1:
			company_name = st.text_input("Company Name", placeholder="Acme Corp")
			homepage_url = st.text_input("Homepage URL", placeholder="https://www.acme.com")
		with col2:
			sic = st.text_input("Primary SIC Industry Classification", placeholder="7372")
			business_desc = st.text_area(
				"Brief Business Description",
				placeholder="Describe core products/services and customers",
				height=120,
			)

		submit = st.form_submit_button("Find Comparables 🚀")

	if not submit:
		st.stop()

	# Basic input validation
	missing = []
	for label, val in [
		("OpenAI API Key", keys.get("openai_api_key")),
		("Finnhub API Key", keys.get("finnhub_api_key")),
		("Company Name", company_name),
		("Homepage URL", homepage_url),
		("Primary SIC Industry Classification", sic),
		("Business Description", business_desc),
	]:
		if not val:
			missing.append(label)
	if missing:
		st.error(f"Missing required fields: {', '.join(missing)}")
		st.stop()

	target = {
		"name": company_name.strip(),
		"url": homepage_url.strip(),
		"business_description": business_desc.strip(),
		"sic": (sic or "").strip(),
	}

	with st.spinner("Searching and synthesizing comparable companies..."):
		try:
			T0 = time.time()
			finder = ComparableFinder(
				openai_api_key=keys["openai_api_key"], finnhub_api_key=keys["finnhub_api_key"]
			)
			# Synchronous pipeline
			raw_output = finder.find_comparables(
				target_name=target["name"],
				target_url=target["url"],
				target_description=target["business_description"],
				target_sic=target["sic"],
			)
			st.write("Stage: find_comparables (LLM+Finnhub)", round(time.time() - T0, 2)); T0 = time.time()
			logger.info("find_comparables completed for %s", target["name"])
		except Exception as e:
			st.error(f"Search/LLM failed: {e}")
			logger.exception("find_comparables failed")
			st.stop()

	try:
		T0 = time.time()
		validated = validate_and_normalize_output(raw_output)
		st.write("Stage: validate_and_normalize_output", round(time.time() - T0, 2)); T0 = time.time()
		logger.info("Validation completed; items=%s", len(validated.get("comparables", [])))
	except Exception as e:
		st.error(f"Malformed or incomplete output: {e}")
		logger.exception("Validation failed")
		st.stop()

	try:
		# Ground tickers/exchanges first (cheap verification)
		T0 = time.time()
		grounded = ground_ticker_exchange(validated["comparables"]) if isinstance(validated, dict) else {"comparables": validated}
		st.write("Stage: ground_ticker_exchange", round(time.time() - T0, 2)); T0 = time.time()
		logger.info("Grounding completed; items=%s", len(grounded.get("comparables", [])))
		# Enforce US SEC registrants only
		us_only = filter_us_sec_registrants(grounded["comparables"]) if isinstance(grounded, dict) else {"comparables": grounded}
		st.write("Stage: filter_us_sec_registrants", round(time.time() - T0, 2)); T0 = time.time()
		if not us_only.get("comparables"):
			st.error("No U.S. SEC-registered public companies found among results.")
			st.stop()
		# Enforce active U.S. listings via Alpha Vantage
		av_key = keys.get("alphavantage_api_key") or ""
		active_us = filter_us_active_listed(
			comparables=us_only["comparables"],
			alpha_vantage_api_key=av_key,
			allowed_exchanges=["NYSE", "NASDAQ", "AMEX"],
		)
		st.write("Stage: filter_us_active_listed (Alpha Vantage)", round(time.time() - T0, 2)); T0 = time.time()
		if not active_us.get("comparables"):
			st.error("No active U.S. listed companies found (Alpha Vantage).")
			st.stop()
		# Compute baseline similarity BEFORE SEC enrichment (fast path)
		scored0 = compute_similarity_scores(
			target_description=target["business_description"],
			comparables=active_us["comparables"],
			openai_api_key=keys["openai_api_key"],
		)
		st.write("Stage: compute_similarity_scores", round(time.time() - T0, 2)); T0 = time.time()
		logger.info("Scoring completed; items=%s", len(scored0.get("comparables", [])))
		scored0 = enrich_similarity_scores(
			target_description=target["business_description"],
			target_sic=target["sic"],
			comparables=scored0["comparables"],
		)
		st.write("Stage: enrich_similarity_scores", round(time.time() - T0, 2)); T0 = time.time()
		scored0 = normalize_similarity_scores(scored0["comparables"])
		st.write("Stage: normalize_similarity_scores", round(time.time() - T0, 2)); T0 = time.time()
		ordered = sorted(scored0["comparables"], key=lambda c: c.get("similarity_score", 0.0), reverse=True)
		st.write("Stage: sort_by_similarity", round(time.time() - T0, 2)); T0 = time.time()
		logger.info("Ranking completed; items=%s", len(ordered))
		# Enrich only top-k with SEC (expensive); keep k small for responsiveness
		top_k = min(6, len(ordered))
		top_for_sec = ordered[:top_k]
		sec_enriched = enrich_with_sec(
			target_name=target["name"],
			comparables=top_for_sec,
			openai_api_key=keys["openai_api_key"],
			default_target_description=target["business_description"],
		)
		st.write("Stage: enrich_with_sec(top_k)", round(time.time() - T0, 2)); T0 = time.time()
		logger.info("SEC enrichment completed; enriched_top=%s", len(sec_enriched.get("comparables", [])))
		# Merge SEC summaries back into top_k while preserving scores
		enriched_map = { (c.get("name") or "") + "|" + (c.get("ticker") or ""): c for c in sec_enriched["comparables"] }
		def _key(c: Dict[str, Any]) -> str:
			return (c.get("name") or "") + "|" + (c.get("ticker") or "")
		merged: List[Dict[str, Any]] = []
		for c in ordered:
			if _key(c) in enriched_map:
				m = dict(c)
				sec_c = enriched_map[_key(c)]
				if sec_c.get("sec_summary"):
					m["sec_summary"] = sec_c["sec_summary"]
				merged.append(m)
			else:
				merged.append(c)
		st.write("Stage: merge_sec_summaries", round(time.time() - T0, 2)); T0 = time.time()
		# Add explainability notes
		ranked = add_reasoning_notes(
			target_description=sec_enriched["target_description"] or target["business_description"],
			target_name=target["name"],
			comparables=merged,
		)
		st.write("Stage: add_reasoning_notes", round(time.time() - T0, 2))
	except Exception as e:
		st.warning(f"Similarity scoring issue (using fallback if possible): {e}")
		ranked = {"comparables": validated["comparables"]}

	comparables: List[Dict[str, Any]] = ranked["comparables"]
	# Deduplicate parent/subsidiary: drop entries whose description mentions the target name
	try:
		name_l = target["name"].lower()
		comparables = [
			c for c in comparables
			if name_l not in ((c.get("business_activity") or "").lower())
		]
	except Exception:
		pass
	if len(comparables) < 3:
		st.error("Fewer than 3 high-quality comparable companies were found. Try refining the description or SIC.")
		st.stop()
	if len(comparables) > 10:
		comparables = comparables[:10]

	st.success(f"Found {len(comparables)} comparable companies.")
	st.subheader("Comparable Companies")
	import pandas as pd
	df = pd.DataFrame(comparables)
	# Header row (text with columns)
	hcols = st.columns([1.8, 3.0, 1.4, 1.2, 3.0, 2.0, 2.0, 1.0])
	hcols[0].markdown("**Name**")
	hcols[1].markdown("**URL**")
	hcols[2].markdown("**Exchange**")
	hcols[3].markdown("**Ticker**")
	hcols[4].markdown("**Business Activity**")
	hcols[5].markdown("**Customer Segment**")
	hcols[6].markdown("**SIC Industry**")
	hcols[7].markdown("**Similarity**")
	# Rows
	for c in comparables:
		cols = st.columns([1.8, 3.0, 1.4, 1.2, 3.0, 2.0, 2.0, 1.0])
		name = c.get("name") or ""
		url = c.get("url") or ""
		exchange = c.get("exchange") or ""
		ticker = c.get("ticker") or ""
		activity = c.get("business_activity") or ""
		segment = c.get("customer_segment") or ""
		industry = c.get("SIC_industry") or ""
		score_val = c.get("similarity_score")
		score_txt = f"{float(score_val):.2f}" if isinstance(score_val, (int, float)) else "-"
		cols[0].markdown(name)
		cols[1].markdown(f"[{url}]({url})" if url else "-")
		cols[2].markdown(exchange or "-")
		cols[3].markdown(ticker or "-")
		cols[4].markdown(activity or "-")
		cols[5].markdown(segment or "-")
		cols[6].markdown(industry or "-")
		cols[7].markdown(score_txt)

	# Download results as CSV/Parquet
	with st.expander("Download as CSV or Parquet"):
		import io
		csv_bytes = df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="Download CSV",
			data=csv_bytes,
			file_name="comparables.csv",
			mime="text/csv",
		)
		try:
			buf = io.BytesIO()
			df.to_parquet(buf, index=False)
			st.download_button(
				label="Download Parquet",
				data=buf.getvalue(),
				file_name="comparables.parquet",
				mime="application/octet-stream",
			)
		except Exception:
			st.info("Parquet requires 'pyarrow'. Run: pip install pyarrow")


if __name__ == "__main__":
	main()


