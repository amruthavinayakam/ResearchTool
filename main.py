import os
from typing import List, Dict, Any

import streamlit as st
import logging
import re

# Configure logging once
if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

from validation import (
	fetch_companies_by_sic,
	map_sic_with_llm,
	filter_companies_with_llm,
	enrich_company_details_with_llm,
	fetch_serp_peers,
)


st.set_page_config(page_title="Comparable Companies Finder", page_icon="📈", layout="wide")


def apply_dark_theme() -> None:
	st.markdown(
		"""
		<style>
		:root, .stApp {
			color-scheme: dark;
		}
		body {
			background-color: #0b0d12;
			color: #f1f5f9;
		}
		.stApp {
			background-color: #0b0d12;
			color: #f1f5f9;
		}
		header, footer {
			background: transparent;
		}
		section[data-testid="stSidebar"] {
			background-color: #111827 !important;
			color: #f1f5f9 !important;
			border-right: 1px solid #1f2937;
		}
		section[data-testid="stSidebar"] * {
			color: #f1f5f9 !important;
		}
		h1, h2, h3, h4, h5, h6, label, p, span {
			color: #f1f5f9 !important;
		}
		input, textarea, select {
			background-color: #1f2937 !important;
			color: #f1f5f9 !important;
			border: 1px solid #374151 !important;
			border-radius: 6px !important;
		}
		button, .stButton button {
			background-color: #2563eb !important;
			color: #f8fafc !important;
			border: none !important;
			border-radius: 6px !important;
		}
		.stForm {
			background-color: #111827;
			padding: 24px 28px;
			border-radius: 16px;
			border: 1px solid #1f2937;
		}
		[data-testid="stDataFrame"] {
			background-color: #0b0d12 !important;
			color: #f1f5f9 !important;
		}
		[data-testid="stDataFrame"] div {
			color: #f1f5f9 !important;
		}
		[data-testid="stTable"] {
			color: #f1f5f9 !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def init_sidebar_keys() -> Dict[str, str]:
	with st.sidebar:
		st.header("API Keys")
		env_openai = os.getenv("OPENAI_API_KEY") or ""
		env_serp = os.getenv("SERPAPI_API_KEY") or ""
		if env_openai:
			st.caption("Using OPENAI_API_KEY from environment.")
			openai_api_key = env_openai
		else:
			openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
		if env_serp:
			st.caption("Using SERPAPI_API_KEY from environment.")
			serp_api_key = env_serp
		else:
			serp_api_key = st.text_input("SerpAPI Key (optional)", type="password", key="serpapi_api_key")
		st.caption("Keys are optional but enable SIC mapping (OpenAI) and external peer discovery (SerpAPI).")
		return {
			"openai_api_key": openai_api_key,
			"serpapi_api_key": serp_api_key,
		}


def main() -> None:
	apply_dark_theme()
	st.title("Comparable Public Companies Generator")
	st.write(
		"Provide target company details. The app will use OpenAI to propose comparable publicly traded companies."
	)

	keys = init_sidebar_keys()

	with st.form("company_form"):
		col1, col2 = st.columns(2)
		with col1:
			company_name = st.text_input("Company Name", placeholder="Acme Corp")
			homepage_url = st.text_input("Homepage URL", placeholder="https://www.acme.com")
		with col2:
			sic = st.text_input("Primary SIC Industry Classification", placeholder="Research and Consulting Services")
			business_desc = st.text_area(
				"Brief Business Description",
				placeholder="Describe core products/services and customers",
				height=120,
			)

		submit = st.form_submit_button("Find Publically Traded Comparables")

	if not submit:
		st.stop()

	# Basic input validation
	missing = []
	for label, val in [
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
	if not re.fullmatch(r"\s*\d{4}\s*", sic or "") and not keys.get("openai_api_key"):
		st.error("OpenAI API Key is required to map SIC descriptions.")
		st.stop()
	openai_key = keys.get("openai_api_key") or ""
	serp_key = keys.get("serpapi_api_key") or ""

	with st.spinner("Resolving SIC and fetching companies from SEC..."):
		try:
			# Resolve SIC (code or best matching name)
			resolved_match: Dict[str, Any] = {}
			sic_list = [part.strip() for part in (sic or "").split(",") if part.strip()]
			numeric_matches = [code for code in sic_list if re.fullmatch(r"\d{4}", code)]
			if numeric_matches:
				code = numeric_matches[0]
				resolved_match = {"sic_code": code, "sic_name": "-", "confidence": 1.0}
				st.success(f"Resolved SIC (manual): {code} — -")
			else:
				resolved_match = map_sic_with_llm(user_input=sic, openai_api_key=openai_key)
				st.success(
					f"Resolved SIC (LLM): {resolved_match.get('sic_code')} — {resolved_match.get('sic_name') or '-'} "
					f"(confidence={resolved_match.get('confidence')})"
				)
			if not resolved_match:
				st.error("Unable to resolve any SIC code.")
				st.stop()
			code = (resolved_match.get("sic_code") or "").strip()
			if not code:
				st.error("Resolved SIC code is empty.")
				st.stop()
			logger.info("main: fetching companies for SIC=%s", code)
			companies = fetch_companies_by_sic(
				sic_code=code,
				start=0,
				count=40,
				owner="include",
			)
			if not companies:
				st.warning("No companies found for this SIC on SEC browse-EDGAR.")
				st.stop()
			# Deduplicate by CIK/ticker while preserving order
			unique_companies: List[Dict[str, Any]] = []
			seen = set()
			for raw in companies:
				entry = dict(raw)
				entry["_sic_code"] = code
				entry["_sic_name"] = resolved_match.get("sic_name") or "-"
				key = (entry.get("cik") or "").strip()
				if not key:
					key = f"{(entry.get('ticker') or '').upper()}::{(entry.get('exchange') or '').upper()}"
				if not key:
					key = (entry.get("name") or "").strip().lower()
				if key in seen:
					continue
				seen.add(key)
				unique_companies.append(entry)
			logger.info("main: aggregated %s unique companies for SIC %s", len(unique_companies), code)
			serp_candidates = []
			if serp_key and company_name:
				try:
					serp_candidates = fetch_serp_peers(
						target_name=company_name,
						serp_api_key=serp_key,
						max_results=10,
					)
				except Exception as exc:
					logger.exception("SerpAPI peer discovery failed: %s", exc)
					st.warning("SerpAPI lookup failed; continuing with SEC results only.")
			# Map to required comparable fields
			comparables = []
			used_keys = set()
			for c in unique_companies:
				sic_label = c.get("_sic_name") or "-"
				if sic_label and sic_label != "-":
					sic_display = f"{c.get('_sic_code')} — {sic_label}"
				else:
					sic_display = c.get("_sic_code") or "unknown"
				entry = {
					"name": c.get("name") or "",
					"url": c.get("url") or "",
					"exchange": c.get("exchange") or "unknown",
					"ticker": c.get("ticker") or "unknown",
					"business_activity": "unknown",
					"customer_segment": "unknown",
					"SIC_industry": sic_display,
				}
				key = ((entry["name"] or "").lower(), entry["url"])
				if key[0] or key[1]:
					used_keys.add(key)
				comparables.append(entry)
			for serp in serp_candidates:
				name = (serp.get("name") or "").strip()
				url = (serp.get("url") or "").strip()
				key = (name.lower(), url)
				if key in used_keys:
					continue
				if key[0] or key[1]:
					used_keys.add(key)
				comparables.append(
					{
						"name": name or (serp.get("title") or url),
						"url": url,
						"exchange": "unknown",
						"ticker": "unknown",
						"business_activity": "unknown",
						"customer_segment": serp.get("snippet") or "unknown",
						"SIC_industry": "SERP peer result",
					}
				)
			filtered_comparables = comparables
			if openai_key:
				try:
					filtered = filter_companies_with_llm(
						target_name=company_name,
						target_description=business_desc,
						companies=comparables,
						openai_api_key=openai_key,
					)
					if filtered:
						filtered_comparables = filtered
					else:
						st.warning("LLM filtering removed all companies. Showing unfiltered results.")
				except Exception as exc:
					logger.exception("LLM filtering failed: %s", exc)
					st.warning("LLM filtering failed; showing unfiltered results.")
				try:
					filtered_comparables = enrich_company_details_with_llm(
						target_name=company_name,
						target_description=business_desc,
						comparables=filtered_comparables,
						openai_api_key=openai_key,
					)
				except Exception as exc:
					logger.exception("LLM enrichment failed: %s", exc)
					st.warning("LLM enrichment failed; some fields may remain 'unknown'.")
			else:
				st.info("Provide an OpenAI API key to enable LLM-based relevance filtering.")
			if not filtered_comparables:
				st.warning("No comparable companies remain after filtering.")
				st.stop()
			sic_display = resolved_match.get("sic_code") if isinstance(resolved_match, dict) else None
			st.subheader(f"Comparable Companies (SEC, SIC {sic_display or 'unknown'})")
			import pandas as pd
			df = pd.DataFrame(filtered_comparables, columns=["name","url","exchange","ticker","business_activity","customer_segment","SIC_industry"])
			st.dataframe(df, use_container_width=True)
			# Download buttons
			csv_bytes = df.to_csv(index=False).encode("utf-8")
			st.download_button(label="Download CSV", data=csv_bytes, file_name="comparables.csv", mime="text/csv")
		except Exception as e:
			st.error(f"SEC scrape failed: {e}")
		st.stop()


if __name__ == "__main__":
	main()


