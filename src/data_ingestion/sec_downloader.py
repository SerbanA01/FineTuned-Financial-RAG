# sec_downloader.py

"""
Asynchronous downloader for fetching and processing SEC 10-K and 10-Q filings.

This module provides a complete, asynchronous pipeline to download financial
filings for a given list of company tickers. It handles CIK lookup, searching
the EDGAR database, parsing search results, and extracting clean text from
the primary filing documents (supporting both HTML and PDF formats).

The pipeline is designed for politeness and robustness, incorporating request
delays, concurrent request limiting, and exponential backoff for retries.
"""

import asyncio
import json
import os
import re
from io import BytesIO
from typing import List, Dict, Optional

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# --- CONFIGURATION ---

# Standard headers for SEC requests. The User-Agent is crucial for responsible scraping,
# identifying the purpose of the script and providing a contact email.
HEADERS = {
	"User-Agent": "University_Project_Financial_LLM (muie.restante@gmail.com)",
	"Accept-Encoding": "gzip, deflate",
	"Host": "www.sec.gov"
}
SEC_BASE_URL = "https://www.sec.gov"
# The SEC provides a JSON file mapping all tickers to their CIKs, which is more efficient than individual lookups.
CIK_LOOKUP_URL = f"{SEC_BASE_URL}/files/company_tickers.json"
# The base URL for the EDGAR search interface.
SEARCH_URL = f"{SEC_BASE_URL}/cgi-bin/browse-edgar"
# Directory where the raw, extracted text from filings will be saved.
DATA_DIR = "sec_filings_raw"
# A mandatory delay between requests to avoid overwhelming the SEC's servers.
REQUEST_DELAY = 1.0
# Limits the number of concurrent network requests to prevent being rate-limited.
MAX_CONCURRENT_REQUESTS = 5

# --- HELPER FUNCTIONS ---

async def _async_get(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, retries: int = 3) -> Optional[bytes]:
	"""
	Performs a rate-limited, asynchronous GET request with retries and error handling.

	Args:
		session: The `aiohttp.ClientSession` to use for the request.
		url: The URL to fetch.
		semaphore: An `asyncio.Semaphore` to limit concurrent requests.
		retries: The maximum number of times to retry the request.

	Returns:
		The raw content of the response as bytes if successful, otherwise None.
	"""
	await asyncio.sleep(REQUEST_DELAY)  # Enforce a respectful delay before every request.
	for attempt in range(retries):
		async with semaphore:  # Acquire the semaphore to limit concurrency.
			try:
				async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
					# Handle the SEC's rate limiting response (HTTP 429).
					if resp.status == 429:
						wait_time = 2 ** (attempt + 4) # Use a longer backoff for rate limiting.
						print(f"Rate limited. Retrying in {wait_time}s...")
						await asyncio.sleep(wait_time)
						continue
					resp.raise_for_status()  # Raises an exception for other 4xx/5xx status codes.
					return await resp.read()
			except Exception as e:
				print(f"Request failed for {url} (attempt {attempt+1}/{retries}): {e}")
				if attempt == retries - 1:
					return None
				# Exponential backoff for transient network errors.
				await asyncio.sleep(2 ** (attempt + 1))
	return None

def _clean_sec_text(text: str) -> str:
	"""
	Applies basic text cleaning to raw content extracted from SEC filings.

	This function removes common artifacts like URLs, boilerplate SEC headers,
	and navigational elements like the Table of Contents to improve the quality
	of the downstream text.

	Args:
		text: The raw text extracted from a filing.

	Returns:
		A cleaned version of the text.
	"""
	# Remove any URLs.
	text = re.sub(r'http[s]?://\S+', '', text)
	# Remove the standard SEC header boilerplate.
	text = re.sub(r"UNITED STATES SECURITIES.?Washington, D.C.\s\d+", "", text, flags=re.DOTALL)
	# Remove the Table of Contents section, which is typically noise.
	text = re.sub(r"Table of Contents.*?(?=Item\s+1)", "", text, flags=re.DOTALL | re.IGNORECASE)
	# Normalize whitespace, remove non-breaking spaces, and strip leading/trailing spaces.
	return re.sub(r'\n+', '\n', text.replace("\xa0", " ")).strip()

async def _extract_text_from_doc(session: aiohttp.ClientSession, doc_url: str, semaphore: asyncio.Semaphore) -> str:
	"""
	Downloads a document and extracts its text, handling both HTML and PDF formats.

	Args:
		session: The `aiohttp.ClientSession` for the download.
		doc_url: The URL of the filing document.
		semaphore: The semaphore for rate limiting.

	Returns:
		The cleaned, extracted text as a string, or an empty string on failure.
	"""
	raw_content = await _async_get(session, doc_url, semaphore)
	if not raw_content:
		return ""
	try:
		# Process PDF files using PyPDF2.
		if doc_url.endswith(".pdf"):
			reader = PdfReader(BytesIO(raw_content))
			text = "\n".join(p.extract_text() or "" for p in reader.pages)
		# Process HTML files using BeautifulSoup.
		else:
			soup = BeautifulSoup(raw_content, "html.parser")
			text = soup.get_text()
		return _clean_sec_text(text)
	except Exception as e:
		print(f"Error extracting text from {doc_url}: {e}")
		return ""

# --- CORE SEC FILING PROCESSING ---

async def download_cik_lookup(session: aiohttp.ClientSession) -> Dict:
	"""
	Downloads and parses the SEC's official ticker-to-CIK mapping file.

	Args:
		session: The `aiohttp.ClientSession` to use for the request.

	Returns:
		A dictionary containing the CIK lookup data, or an empty dictionary on failure.
	"""
	try:
		async with session.get(CIK_LOOKUP_URL, headers=HEADERS) as resp:
			resp.raise_for_status()
			# Sanity check to ensure we received a JSON file.
			content_type = resp.headers.get("Content-Type", "")
			if "application/json" not in content_type:
				raise Exception(f"Unexpected content-type: {content_type}")
			print("Successfully downloaded CIK lookup file.")
			return await resp.json()
	except Exception as e:
		print(f"Failed to fetch CIK lookup: {e}")
		return {}

def get_cik_from_lookup(ticker: str, lookup_data: dict) -> Optional[str]:
	"""
	Finds a company's CIK from the pre-downloaded lookup data.

	Args:
		ticker: The company's stock ticker symbol.
		lookup_data: The dictionary of CIK data from `download_cik_lookup`.

	Returns:
		A zero-padded 10-digit CIK string if found, otherwise None.
	"""
	for entry in lookup_data.values():
		if entry["ticker"].lower() == ticker.lower():
			# CIKs must be 10 digits long for EDGAR URLs, so we left-pad with zeros.
			return str(entry["cik_str"]).zfill(10)
	return None

async def get_filing_urls(session: aiohttp.ClientSession, cik: str, filing_type: str, max_docs: int) -> List[Dict[str, str]]:
	"""
	Searches the EDGAR database for recent filings for a given CIK and filing type.

	Args:
		session: The `aiohttp.ClientSession` for the search request.
		cik: The company's 10-digit CIK number.
		filing_type: The type of filing to search for (e.g., "10-K").
		max_docs: The maximum number of filing URLs to return.

	Returns:
		A list of dictionaries, each containing the `detail_url` and `date` for a filing.
	"""
	try:
		params = {
			"action": "getcompany", "CIK": cik, "type": filing_type,
			"owner": "exclude", "count": "100" # Request a large number and then slice locally.
		}
		async with session.get(SEARCH_URL, params=params, headers=HEADERS) as resp:
			resp.raise_for_status()
			content = await resp.text()
			soup = BeautifulSoup(content, "html.parser")
			table = soup.find("table", class_="tableFile2")
			filings = []
			if table:
				# Iterate through rows in the search results table, skipping the header.
				for row in table.find_all("tr")[1:]:
					cols = row.find_all("td")
					link = cols[1].find("a") if len(cols) > 1 else None
					date = cols[3].text.strip() if len(cols) > 3 else None
					if link and date:
						# The link is to a detail page, not the final document.
						filings.append({"detail_url": f"{SEC_BASE_URL}{link['href']}", "date": date})
					if len(filings) >= max_docs:
						break
			return filings
	except Exception as e:
		print(f"Error fetching filing URLs for CIK {cik}: {e}")
		return []

async def _process_filing(session: aiohttp.ClientSession, filing: Dict, ticker: str, filing_type: str, semaphore: asyncio.Semaphore) -> Optional[Dict]:
	"""
	Processes a single filing's detail page to find and extract text from the main document.

	Args:
		session: The `aiohttp.ClientSession` to use.
		filing: A dictionary containing the filing's `detail_url`.
		ticker: The company ticker symbol.
		filing_type: The type of filing being processed (e.g., "10-K").
		semaphore: The semaphore for rate limiting.

	Returns:
		A dictionary of the processed document data if successful, otherwise None.
	"""
	raw_detail_page = await _async_get(session, filing["detail_url"], semaphore)
	if not raw_detail_page:
		return None
	soup = BeautifulSoup(raw_detail_page, "html.parser")
	table = soup.find("table", class_="tableFile")
	if not table:
		return None

	# The detail page lists multiple files; we need to find the main one.
	for row in table.find_all("tr")[1:]:
		cols = row.find_all("td")
		doc_type = cols[3].text.strip().lower() if len(cols) > 3 else ""
		href_tag = cols[2].find("a") if len(cols) > 2 else None

		if not href_tag:
			continue

		href = href_tag["href"]
		filename = href.split("/")[-1].lower()

		# These heuristics are key to isolating the primary filing document and ignoring
		# ancillary files like exhibits, XML/XBRL data, or graphical supplements.
		is_correct_type = doc_type == filing_type.lower()
		is_html = filename.endswith((".htm", ".html"))
		is_not_exhibit = not any(x in filename for x in ["xbrl", "xml", "ex-", "exhibit", "form", "index"])

		if is_correct_type and is_html and is_not_exhibit:
			# Clean the URL to get a direct link to the document.
			doc_url = f"{SEC_BASE_URL}{href}".replace("/ix?doc=", "")
			text = await _extract_text_from_doc(session, doc_url, semaphore)
			if not text:
				continue # Skip if text extraction failed.

			return {
				"ticker": ticker,
				"date": filing["date"],
				"text": text,
				"url": doc_url,
				"filing_type": filing_type
			}
	return None

async def _process_company(session: aiohttp.ClientSession, ticker: str, cik: str, semaphore: asyncio.Semaphore, filing_type: str, max_docs: int):
	"""
	Orchestrates the complete download and processing workflow for a single company and filing type.

	Args:
		session: The `aiohttp.ClientSession` to use.
		ticker: The company ticker symbol.
		cik: The company CIK number.
		semaphore: The semaphore for rate limiting.
		filing_type: The type of filing to download (e.g., "10-K").
		max_docs: The maximum number of documents of this type to download.
	"""
	filing_dir = os.path.join(DATA_DIR, ticker, filing_type)
	# This check makes the pipeline idempotent; it can be re-run without re-downloading existing data.
	if os.path.exists(filing_dir) and os.listdir(filing_dir):
		print(f"Skipping existing data for: {ticker} {filing_type}")
		return
	filings = await get_filing_urls(session, cik, filing_type, max_docs)
	if not filings:
		print(f"No '{filing_type}' filings found for {ticker}")
		return

	for filing in filings:
		path = os.path.join(filing_dir, f"{filing['date']}.json")
		# Skip files that have already been saved within this run.
		if os.path.exists(path):
			continue

		doc_data = await _process_filing(session, filing, ticker, filing_type, semaphore)
		if doc_data:
			os.makedirs(os.path.dirname(path), exist_ok=True)
			# Asynchronously write the final JSON data to a file.
			async with aiofiles.open(path, "w", encoding="utf-8") as f:
				await f.write(json.dumps(doc_data, indent=2))
			print(f"Successfully saved: {path}")

async def run_pipeline(tickers: List[str], max_10k: int = 5, max_10q: int = 15):
	"""
	Main entry point for the SEC filing download pipeline.

	This function orchestrates the entire process of downloading 10-K and 10-Q
	filings for a list of company tickers.

	Args:
		tickers: A list of stock ticker symbols to process.
		max_10k: The maximum number of the most recent 10-K filings to download per ticker.
		max_10q: The maximum number of the most recent 10-Q filings to download per ticker.
	"""
	# The semaphore is created once and passed down to control all concurrent requests.
	semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
	async with aiohttp.ClientSession() as session:
		cik_lookup = await download_cik_lookup(session)
		if not cik_lookup:
			print("Aborting: Failed to download CIK lookup data.")
			return

		# Create a list of asynchronous tasks to be executed concurrently.
		tasks = []
		for ticker in tickers:
			cik = get_cik_from_lookup(ticker, cik_lookup)
			if not cik:
				print(f"CIK not found for ticker: {ticker}")
				continue

			print(f"Queueing downloads for {ticker} (CIK: {cik})")
			# Queue up tasks for both 10-K and 10-Q filings for the company.
			tasks.append(_process_company(session, ticker, cik, semaphore, "10-K", max_10k))
			tasks.append(_process_company(session, ticker, cik, semaphore, "10-Q", max_10q))

		# `asyncio.gather` runs all the queued tasks concurrently.
		await asyncio.gather(*tasks)
		print("\n--- SEC filing download pipeline finished. ---")