Comparable Companies Finder
===========================

This Streamlit app discovers publicly traded comparables for a target company by combining:

- SEC browse-EDGAR scraping for the target's Standard Industrial Classification (SIC)
- Optional OpenAI calls to resolve free-text industry descriptions, filter noisy matches, and enrich missing fields
- Optional SerpAPI Google searches to surface additional peer candidates

The UI guides you from data entry through candidate discovery, filtering, enrichment, and CSV export.

Quick Start
-----------

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**

   ```bash
   streamlit run main.py
   ```

3. **Provide API keys in the sidebar**

   - `OPENAI_API_KEY` (required if you enter a non-numeric SIC description or want LLM filtering/enrichment)
   - `SERPAPI_API_KEY` (optional; adds Google-based peer discovery)

   You can either set the environment variables beforehand or paste them into the text boxes when prompted.

Feature Overview
----------------

- **Single SIC resolution**  
  - Numeric SIC input is accepted directly.  
  - Free-text descriptions are resolved to one 4-digit code via `map_sic_with_llm`.

- **SEC company scraping**  
  - `fetch_companies_by_sic` walks the browse-EDGAR pages, enriches each CIK with ticker/exchange/URL, and keeps only U.S.-listed companies.

- **SerpAPI peer signals (optional)**  
  - `fetch_serp_peers` runs “`<target> publicly traded competitors`” and “`<target> publicly traded peers`” Google searches, merging the results with the SEC list.

- **LLM filtering and enrichment (optional)**  
  - `filter_companies_with_llm` removes candidates that are not truly comparable based on the target description.  
  - `enrich_company_details_with_llm` fills in missing websites, business activities, and customer segments.

- **Exportable results**  
  - Streamlit displays the final table and offers a CSV download (`name`, `url`, `exchange`, `ticker`, `business_activity`, `customer_segment`, `SIC_industry`).

Environment Variables
---------------------

| Variable            | Purpose                                                               |
|---------------------|------------------------------------------------------------------------|
| `OPENAI_API_KEY`    | Required for SIC mapping, LLM filtering, and enrichment steps          |
| `SERPAPI_API_KEY`   | Optional; enables Google/SerpAPI peer discovery                        |

Either export these before launching Streamlit or paste them at runtime in the sidebar.

Operational Notes
-----------------

- The SEC site occasionally times out or rate-limits; if you see `Read timed out`, rerun or increase the timeout in `fetch_companies_by_sic`.
- Streamlit warns that `use_container_width` will be deprecated—update to `width='stretch'` before 2026.
- Logging is already configured; watch the terminal to trace which functions run during a session.

License
-------

MIT (or align with your project’s chosen license).
