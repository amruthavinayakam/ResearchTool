<<<<<<< HEAD
# ResearchTool
=======
Comparable Companies Finder (Streamlit + LangChain)
==================================================

Find 3–10 comparable publicly traded companies for a target company using SerpAPI and OpenAI (GPT-4o-mini). Includes structured JSON output and a similarity validation step with embeddings.

Setup
-----

1. Create a virtual environment (recommended) and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run main.py
```

3. In the app sidebar, provide your OpenAI API Key and Finnhub API Key (SerpAPI optional/legacy).

Output Schema
-------------
Each comparable has the following exact keys:
- name
- url
- exchange
- ticker
- business_activity
- customer_segment
- SIC_industry

SIC_industry details:
- This field returns the SIC industry group name(s), not numeric codes.
- If multiple groups apply, the names are comma-separated in a single string.

Notes
-----
- Uses LangChain for orchestrating OpenAI calls and Finnhub peers/profile for candidate discovery.
- Similarity scoring uses `text-embedding-3-small` with cosine similarity, falls back to keyword overlap if embeddings fail.

Pipeline overview
-----------------
1. Extract keywords, industry groups, products, and customer segments from the target description (LLM JSON).
2. Build rich SerpAPI queries using extracted signals (industries, keywords, products, segments, SIC, URL domain).
3. Fetch candidate companies via SerpAPI and deduplicate.
4. Rank candidates by semantic similarity (embeddings) between target description and candidate title/snippet; keep the top candidates.
5. Two-tier LLM:
   - Classifier LLM (cheap) filters to relevant, publicly traded candidates.
   - Formatter LLM (stronger) enriches and outputs the final JSON with required fields.
6. Validate schema, normalize fields, and re-rank/filter by similarity before presenting results.
7. Ground ticker/exchange via Yahoo Finance (quote + autocomplete) to correct or fill missing values.

>>>>>>> 91fd276 (Initial Commit)
