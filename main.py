import os
from typing import List, Dict, Any

import streamlit as st
import logging
import re

# Configure logging once
if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

from validation import fetch_companies_by_sic, fetch_sic_index, map_sic_with_llm, filter_companies_with_llm


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
		"Provide target company details. The app will use SerpAPI and OpenAI to propose comparable publicly traded companies."
	)

	keys = init_sidebar_keys()

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
			code = ""
			sic_name = "-"
			if re.fullmatch(r"\s*\d{4}\s*", sic or ""):
				code = (sic or "").strip()
				# Map code->name from SEC list (no hardcoding)
				try:
					entries = fetch_sic_index()
					for e in entries:
						if e.get("code") == code:
							sic_name = e.get("name") or "-"
							break
				except Exception:
					sic_name = "-"
				st.success(f"Resolved SIC: {code} — {sic_name}")
			else:
				llm_res = map_sic_with_llm(user_input=sic, openai_api_key=openai_key)
				code = (llm_res.get("sic_code") or "").strip()
				sic_name = llm_res.get("sic_name") or "-"
				conf = llm_res.get("confidence")
				st.success(f"Resolved SIC (LLM): {code} — {sic_name} (confidence={conf})")
			# Fetch companies for resolved SIC
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
			logger.info("main: fetched %s companies for SIC %s", len(companies), code)
			# Map to required comparable fields
			comparables = []
			for c in companies:
				comparables.append(
					{
						"name": c.get("name") or "",
						"url": c.get("url") or "",
						"exchange": c.get("exchange") or "unknown",
						"ticker": c.get("ticker") or "unknown",
						"business_activity": "unknown",
						"customer_segment": "unknown",
						"SIC_industry": sic_name,
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
			else:
				st.info("Provide an OpenAI API key to enable LLM-based relevance filtering.")
			if not filtered_comparables:
				st.warning("No comparable companies remain after filtering.")
				st.stop()
			st.subheader(f"Comparable Companies (SEC, SIC {code})")
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


