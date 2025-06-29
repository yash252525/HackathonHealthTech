import json
import requests
import spacy
import openai

# Set your OpenAI API key
openai.api_key = "###########"

# Set your PubMed API key
pubmed_api_key = "#############"

# Load the pre-trained spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    exit(1)


def clean_json_response(response_str):
    """Removes markdown code fences from a string, if present."""
    response_str = response_str.strip()
    if response_str.startswith("```"):
        lines = response_str.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return response_str


def get_openai_entities(text):
    """
    Uses OpenAI to perform named entity recognition on a given text.
    The prompt explicitly asks for standard, biomedical, and scientific entities.
    Returns a list of entities, each with keys 'text' and 'label'.
    """
    prompt = (
        "Extract all named entities from the following text. "
        "Include standard entities (like persons, organizations, locations) as well as biomedical and scientific terms "
        "(such as bacteria names, antibiotics, chemical compounds, and scientific names). "
        "Return them as a JSON list where each entity is an object with keys 'text' and 'label'. "
        "Make sure to include any relevant bacteria, antibiotic, or scientific term. "
        "If no entities are found, return an empty list.\n\n"
        f"Text: \"{text}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "You are an assistant that extracts both general and biomedical/scientific named entities."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message['content']
        answer = clean_json_response(answer)
        try:
            entities = json.loads(answer)
            if isinstance(entities, list):
                return entities
            else:
                print("Unexpected format from OpenAI NER. Expected a JSON list.")
                return []
        except json.JSONDecodeError:
            print("Failed to decode OpenAI response as JSON. Raw response:")
            print(answer)
            return []
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return []


def get_query_entities(query):
    """
    Extract named entities from the query using OpenAI.
    If OpenAI returns no entities, fall back to spaCy.
    """
    entities = get_openai_entities(query)
    if not entities:
        print("OpenAI did not return any entities for the query. Falling back to spaCy...")
        doc = nlp(query)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    return entities


def filter_papers(papers, required_terms):
    """
    Post-filter papers to include those having at least one of the required terms
    in either the title or abstract.
    """
    filtered = []
    for paper in papers:
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        if any(term.lower() in title for term in required_terms) or any(
                term.lower() in abstract for term in required_terms):
            filtered.append(paper)
    return filtered


def fetch_paper_abstract(query, limit=15):
    """
    Fetches research papers from Semantic Scholar.
    For each paper, extracts title and abstract (using spaCy first, then OpenAI NER if needed).
    Returns a list of dictionaries with title, abstract, and named entities.
    """
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,
        'limit': limit,
        'fields': "title,abstract"
    }
    api_key = "##########"
    headers = {"x-api-key": api_key}
    try:
        response = requests.get(search_url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"Semantic Scholar Error: Received status code {response.status_code}")
            print("Response content:", response.text)
            return []
        data = response.json()
        papers = []
        if 'data' not in data or len(data['data']) == 0:
            print("No papers found for the given query (Semantic Scholar).")
            return []
        for paper in data['data']:
            title = paper.get('title', 'No title available')
            abstract = paper.get('abstract', None)
            if abstract:
                doc = nlp(abstract)
                entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
                if not entities:
                    print(f"spaCy did not find entities in abstract for: {title}. Trying OpenAI NER as fallback...")
                    entities = get_openai_entities(abstract)
            else:
                abstract = "No abstract available"
                entities = []
            papers.append({
                'title': title,
                'abstract': abstract,
                'entities': entities
            })
        return papers
    except requests.exceptions.RequestException as e:
        print(f"Semantic Scholar Request failed: {e}")
        return []


def parse_medline(medline_text):
    """
    Parses MEDLINE-formatted text to extract title and abstract.
    This simple parser looks for lines starting with 'TI  - ' and 'AB  - '.
    """
    title_lines = []
    abstract_lines = []
    lines = medline_text.splitlines()
    current_field = None
    for line in lines:
        if line.startswith("TI  - "):
            current_field = "TI"
            title_lines.append(line.replace("TI  - ", "").strip())
        elif line.startswith("AB  - "):
            current_field = "AB"
            abstract_lines.append(line.replace("AB  - ", "").strip())
        elif line.startswith("      ") and current_field == "AB":
            abstract_lines.append(line.strip())
        elif line.startswith("      ") and current_field == "TI":
            title_lines.append(line.strip())
        else:
            current_field = None
    title = " ".join(title_lines) if title_lines else "No title available"
    abstract = " ".join(abstract_lines) if abstract_lines else "No abstract available"
    return {"title": title, "abstract": abstract}


def fetch_pubmed_abstract(query, limit=15):
    """
    Fetches research articles from PubMed.
    Uses ESearch to get a list of PubMed IDs, then retrieves details in MEDLINE format
    via the PubMed endpoint with format=medline and download=y.
    Returns a list of dictionaries with title, abstract, and named entities.
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "api_key": pubmed_api_key
    }
    try:
        esearch_resp = requests.get(esearch_url, params=esearch_params)
        if esearch_resp.status_code != 200:
            print(f"PubMed ESearch Error: Received status code {esearch_resp.status_code}")
            print("Response content:", esearch_resp.text)
            return []
        esearch_data = esearch_resp.json()
        id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            print("No PubMed IDs found for the given query.")
            return []
        papers = []
        for pmid in id_list:
            ctxp_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
            ctxp_params = {
                "id": pmid,
                "format": "medline",
                "download": "y",
                "api_key": pubmed_api_key
            }
            ctxp_resp = requests.get(ctxp_url, params=ctxp_params)
            if ctxp_resp.status_code != 200:
                print(f"PubMed Error for id {pmid}: Received status code {ctxp_resp.status_code}")
                print("Response content:", ctxp_resp.text)
                continue
            medline_text = ctxp_resp.text
            parsed = parse_medline(medline_text)
            doc = nlp(parsed["abstract"])
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
            if not entities:
                entities = get_openai_entities(parsed["abstract"])
            papers.append({
                "title": parsed["title"],
                "abstract": parsed["abstract"],
                "entities": entities
            })
        return papers
    except requests.exceptions.RequestException as e:
        print(f"PubMed Request failed: {e}")
        return []


def generate_combined_abstract(papers):
    """
    Takes a list of papers and combines their abstracts into a single string.
    Then sends the combined string to OpenAI for summary.
    """
    combined_text = " ".join([paper["abstract"] for paper in papers])

    prompt = f"Summarize the following abstracts into a concise paragraph:\n\n{combined_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        summary = response.choices[0].message['content']
        return summary
    except Exception as e:
        print(f"Error in OpenAI generation: {e}")
        return "Unable to generate summary."


def display_results(query_entities, papers):
    """
    Display extracted named entities from the query and research papers.
    """
    print("\nNamed Entities in Query:")
    if query_entities:
        for ent in query_entities:
            print(f"  - {ent['text']} ({ent['label']})")
    else:
        print("No named entities extracted from the query.")

    print("\nFetching papers from Semantic Scholar and PubMed...\n")
    if not papers:
        print("No data available.")
        return

    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract']}\n")
        print("-" * 50)  # Divider between papers

    # Now generate the combined abstract using RAG
    combined_abstract = generate_combined_abstract(papers)
    print("\nGenerated Combined Abstract from Papers:")
    print(combined_abstract)


def main():
    query = "Among Cas9-disrupted loci in human neural stem cells, what fraction of disruption phenotypes were apparent after 4 cell divisions?"

    # Extract named entities from the query using OpenAI (with spaCy fallback)
    query_entities = get_query_entities(query)

    # Create a refined query using an "AND" operator to ensure all entities are included.
    if query_entities:
        refined_query = " AND ".join([str(ent['text']) for ent in query_entities])
    else:
        refined_query = query
    print("Refined Query:", refined_query)

    # Fetch papers from Semantic Scholar
    semantic_papers = fetch_paper_abstract(refined_query)

    # Fetch papers from PubMed
    pubmed_papers = fetch_pubmed_abstract(refined_query)

    # Combine the results from both sources
    combined_papers = semantic_papers + pubmed_papers

    # Post-filter papers to only include those with at least one required term (in title or abstract)
    required_terms = [ent['text'] for ent in query_entities]
    filtered_papers = filter_papers(combined_papers, required_terms)

    # If no results with refined query, fall back to original query for both sources.
    if not filtered_papers:
        print("No results found for the refined query. Trying the original query...")
        semantic_papers = fetch_paper_abstract(query)
        pubmed_papers = fetch_pubmed_abstract(query)
        combined_papers = semantic_papers + pubmed_papers
        filtered_papers = filter_papers(combined_papers, required_terms)

    # Display the results for the single query.
    display_results(query_entities, filtered_papers)


if __name__ == "__main__":
    main()
