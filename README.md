**# global-social-protest
GSP is a database on global social protest from 1851-2023 created by the Global Social Protest Research Working Group at the Arrighi Center for Global Studies (Johns Hopkins University). I contest that their methods are outdated, and a larger, more robust database can be created and analyzed by leveraging LLMs.
**# Global Social Protest (GSP-LLM)

A large-scale, LLM-enhanced database of global social protest events (1851–2023), built from newspaper archives and structured using programmatic AI coding.

This project critiques and extends the methodology of the Global Social Protest (GSP) dataset developed by the Global Social Protest Research Working Group at the Arrighi Center for Global Studies. We argue that traditional hand-coded protest datasets are limited in scale, consistency, and reproducibility — and that modern LLMs enable a more robust, transparent, and extensible approach.

---

## Overview

The system automatically:

* Classifies whether an article describes **social protest**
* Distinguishes vertical vs horizontal conflict
* Identifies whether the event is **current**
* Determines whether the event occurred in **India**
* Extracts structured protest metadata:

  * City
  * Other location
  * Actors
  * Demands
  * Action type

---

## Motivation

Existing protest datasets rely heavily on:

* Manual coding
* Limited source coverage
* Binary or overly coarse classification systems

LLMs allow us to:

* Scale to hundreds of thousands of articles
* Apply consistent rule-based definitions
* Extract structured variables beyond simple protest detection
* Maintain deterministic outputs (temperature = 0)
* Produce reproducible JSON-coded outputs

This project demonstrates how computational social science can leverage foundation models for historical event coding.

---

## Data Source

* **Newspaper**: Times of India (TOI)
* **Accessed via**: ProQuest Historical Newspapers
* **Fields Used**:

  * Title
  * Abstract
  * Publication Date

> Note: Raw ProQuest data is not included due to licensing restrictions.

---

## Coding Framework

Each article is classified into structured fields:

### 1️⃣ SP — Social Protest Score

| Code | Meaning                                                      |
| ---- | ------------------------------------------------------------ |
| 2    | Vertical/Hierarchical protest (subordinate vs superordinate) |
| 9    | Horizontal conflict or individual action                     |
| 1    | No protest / false positive                                  |

Vertical protest examples:

* Workers vs employers
* Citizens vs state
* Students vs administration
* Election boycotts

---

### 2️⃣ CU — Current Event

| Code | Meaning                                       |
| ---- | --------------------------------------------- |
| 2    | Event occurred within 365 days of publication |
| 1    | Event occurred more than 365 days prior       |

---

### 3️⃣ INT — International

| Code | Meaning                      |
| ---- | ---------------------------- |
| 2    | Event occurred in India      |
| 1    | Event occurred outside India |

---

### 4️⃣ Extracted Variables

* `City`
* `Other_Location`
* `Actors`
* `Demands`
* `Action_Type`

All outputs are returned as structured JSON.

# Main Pipeline

Below is the core labeling pipeline used in the project.

---

## Load Data

```python
df = pd.read_csv('big_database_all_ab.csv')
df = df.iloc[:, 2:]
df
```

---

## LLM Coding System

```python
import os
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_instruction = """
You are an expert sociologist and data coder specializing in newspaper analysis. 
Your task is to analyze a newspaper article (Title + Abstract) and code it into specific categories based on strict rules.

Return the result as a valid JSON object.

### CODING RULES:

1. SP (Social Protest) Score:
   - 2 (Yes): Vertical/Hierarchical conflict...
   - 9 (Horizontal/Individual)
   - 1 (No)

2. CU (Current Event):
   - 2 (Yes): Within 365 days
   - 1 (No)

3. INT (International/India):
   - 2 (Yes): Event occurred in India
   - 1 (No)

4. Extraction Fields:
   - City
   - Other_Location
   - Actors
   - Demands
   - Action_Type

Return ONLY a JSON object with keys:
"SP", "CU", "INT", "City", "Other_Location", "Actors", "Demands", "Action_Type".
"""
```

---

## Article Analysis Function

```python
def analyze_article(title, abstract, pub_date):
    user_content = f"""
    Publication Date: {pub_date}
    Title: {title}
    Abstract: {abstract}
    
    Analyze this entry based on the system rules.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error processing article: {e}")
        return None
```

---

## Batch Processing

```python
df['PubDate'] = df['PubDate'].astype(str)

results = []

trial_df = df[:1000]

for index, row in tqdm(trial_df.iterrows(), total=trial_df.shape[0]):
    
    analysis = analyze_article(row['Title'], row['Abstract'], row['PubDate'])
    
    if analysis:
        results.append(analysis)
    else:
        results.append({
            "SP": None, "CU": None, "INT": None, 
            "City": None, "Other_Location": None, 
            "Actors": None, "Demands": None, "Action_Type": None
        })

results_df = pd.DataFrame(results)
final_df = pd.concat([trial_df.reset_index(drop=True), results_df], axis=1)

final_df.to_csv('first_1000_labeled.csv')
```

---

# Reproducibility

## Setup

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Run notebook or script.

---

# Methodological Advantages

* Deterministic classification (temperature = 0)
* Structured JSON enforcement
* Scalable to 100k+ articles
* Transparent rule-based definitions
* Easily auditable prompt
* Extensible to multilingual corpora

---

# Limitations

* Dependent on archive access (ProQuest licensing)
* LLM classification may require periodic validation
* Historical language variation may affect extraction accuracy
* Cost scales with dataset size

---

# Future Directions

* Human validation subset for intercoder reliability comparison
* Embedding-based clustering of protest types
* Temporal modeling of protest frequency
* Cross-newspaper comparative analysis
* Expansion beyond India corpus
* Fine-tuned model for protest detection

---

# Citation

If using this dataset or pipeline, please cite:

> Taneja, R. (2026). Global Social Protest (GSP-LLM): An LLM-Based Protest Event Coding Pipeline.

---

# Contact

Rohan Taneja
Computational Social Science & Applied ML

---

If you're building Washed-level ML systems and also doing historical event modeling — this repo sits exactly at that intersection: structured AI systems applied to messy real-world human data.
