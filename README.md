# 🧠 Biomedical Literature Review Assistant (HealthTech Hackathon Project)

This project is an AI-powered biomedical literature review assistant developed for the **UoB HealthTech AI Hackathon**. It helps researchers and clinicians extract and summarize key scientific entities (e.g., diseases, genes, bacteria) from biomedical text queries by leveraging **OpenAI GPT-3.5**, **spaCy**, **PubMed**, and **Semantic Scholar** APIs.

---

## 🚀 Features

- 🔍 Named Entity Recognition using **OpenAI GPT-3.5** and **spaCy**
- 📄 Fetches and parses papers from:
  - **Semantic Scholar Graph API**
  - **PubMed (MEDLINE format)**
- 🧬 Extracts scientific entities including biomedical terms, diseases, compounds, and general NER
- 🧠 Summarizes abstracts from multiple papers into a **concise research summary**
- 🔄 Fallback mechanism: Uses spaCy if OpenAI fails to extract entities

---

## 🏗️ Architecture & Flow

```
User Query → NER via GPT-3.5 → Refined Query → Search (PubMed + Semantic Scholar) 
→ Parse Abstracts & Entities → RAG-style Summary via OpenAI → Display Results
```

---

## 📦 Requirements

- Python 3.7+
- spaCy (`en_core_web_sm`)
- OpenAI API Key
- PubMed API Key
- Semantic Scholar API Key

---

## 🔧 Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/biomedical-literature-assistant.git
   cd biomedical-literature-assistant
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**  
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Configure API Keys**  
   Open the script and replace:
   ```python
   openai.api_key = "your-openai-key"
   pubmed_api_key = "your-pubmed-key"
   api_key = "your-semantic-scholar-key"
   ```

---

## 🧪 Usage

Run the script directly:

```bash
DataFetch.py
```

Example query:
```
Among Cas9-disrupted loci in human neural stem cells, what fraction of disruption phenotypes were apparent after 4 cell divisions?
```

---

## 📝 Output

- Named Entities extracted from the query
- Titles and abstracts from top research papers
- Automatically generated **summary paragraph** combining the findings

---

## 📚 Tech Stack

- `OpenAI GPT-3.5`
- `spaCy` (NER model)
- `Semantic Scholar Graph API`
- `PubMed E-Utilities & MEDLINE`
- `Python (requests, json, NLP)`

---

## 👥 Team

- **Yash Zore** – Backend Developer, AI/NLP Integration  
*(Solo entry for HealthTech Hackathon @ University of Birmingham)*


## 📬 Contact

📧 yashzore321@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/ysz252525)  
🌍 [GitHub](https://github.com/yash252525)

