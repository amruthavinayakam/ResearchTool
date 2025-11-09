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
	fetch_sic_index,
	map_sic_with_llm,
	filter_companies_with_llm,
	enrich_company_details_with_llm,
)


st.set_page_config(page_title="Comparable Companies Finder", page_icon="📈", layout="wide")


def init_sidebar_keys() -> Dict[str, str]:
	with st.sidebar:
		st.header("API Keys")
		env_openai = os.getenv("OPENAI_API_KEY") or ""
		if env_openai:
			st.caption("Using OPENAI_API_KEY from environment.")
			openai_api_key = env_openai
		else:
			openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
		st.caption("Key is used to map SIC descriptions via OpenAI.")
		return {
			"openai_api_key": openai_api_key,
		}


def main() -> None:
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

	with st.spinner("Resolving SIC and fetching companies from SEC..."):
		try:
			# Resolve SIC (code or best matching name)
			resolved_matches: List[Dict[str, Any]] = []
			sic_list = [part.strip() for part in (sic or "").split(",") if part.strip()]
			numeric_matches = [code for code in sic_list if re.fullmatch(r"\d{4}", code)]
			if numeric_matches:
				try:
					entries = fetch_sic_index()
					name_lookup = {entry.get("code"): entry.get("name") for entry in entries}
				except Exception:
					name_lookup = {}
				for idx, code in enumerate(numeric_matches[:2]):
					resolved_matches.append(
						{
							"sic_code": code,
							"sic_name": name_lookup.get(code) or "-",
							"confidence": 1.0 if idx == 0 else None,
						}
					)
				st.success(
					"Resolved SIC (manual): "
					+ ", ".join(
						f"{match['sic_code']} — {match.get('sic_name') or '-'}"
						for match in resolved_matches
					)
				)
			else:
				llm_res = map_sic_with_llm(user_input=sic, openai_api_key=openai_key)
				matches = llm_res.get("matches") or []
				if not matches:
					raise ValueError("LLM did not return any SIC matches.")
				resolved_matches = matches[:2]
				st.success(
					"Resolved SIC (LLM): "
					+ ", ".join(
						f"{match.get('sic_code')} — {match.get('sic_name') or '-'}"
						for match in resolved_matches
					)
				)
			if not resolved_matches:
				st.error("Unable to resolve any SIC codes.")
				st.stop()
			# Fetch companies for each resolved SIC
			all_companies: List[Dict[str, Any]] = []
			for order, match in enumerate(resolved_matches):
				code = (match.get("sic_code") or "").strip()
				if not code:
					continue
				logger.info("main: fetching companies for SIC=%s (order=%s)", code, order)
				companies = fetch_companies_by_sic(
					sic_code=code,
					start=0,
					count=40,
					owner="include",
				)
				if not companies:
					logger.info("main: no companies found for SIC=%s", code)
					continue
				for entry in companies:
					entry = dict(entry)
					entry["_sic_code"] = code
					entry["_sic_name"] = match.get("sic_name") or "-"
					entry["_sic_order"] = order
					all_companies.append(entry)
			if not all_companies:
				st.warning("No companies found for the resolved SIC codes on SEC browse-EDGAR.")
				st.stop()
			# Deduplicate by CIK/ticker while preserving order (primary code first)
			unique_companies: List[Dict[str, Any]] = []
			seen = set()
			for entry in all_companies:
				key = (entry.get("cik") or "").strip()
				if not key:
					key = f"{(entry.get('ticker') or '').upper()}::{(entry.get('exchange') or '').upper()}"
				if not key:
					key = (entry.get("name") or "").strip().lower()
				if key in seen:
					continue
				seen.add(key)
				unique_companies.append(entry)
			logger.info("main: aggregated %s unique companies across %s SIC codes", len(unique_companies), len(resolved_matches))
			# Map to required comparable fields
			comparables = []
			for c in unique_companies:
				sic_label = c.get("_sic_name") or "-"
				if sic_label and sic_label != "-":
					sic_display = f"{c.get('_sic_code')} — {sic_label}"
				else:
					sic_display = c.get("_sic_code") or "unknown"
				comparables.append(
					{
						"name": c.get("name") or "",
						"url": c.get("url") or "",
						"exchange": c.get("exchange") or "unknown",
						"ticker": c.get("ticker") or "unknown",
						"business_activity": "unknown",
						"customer_segment": "unknown",
						"SIC_industry": sic_display,
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
			sic_codes_display = ", ".join(match.get("sic_code") for match in resolved_matches if match.get("sic_code"))
			st.subheader(f"Comparable Companies (SEC, SIC {sic_codes_display})")
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


