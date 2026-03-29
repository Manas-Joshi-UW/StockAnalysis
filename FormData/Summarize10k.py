import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from typing import List, Dict
load_dotenv()  # reads .env from project root
print(os.getenv("HUGGINGFACE_API_KEY"))
print("Starting OpenAI client...")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

def summarize_10k(text: str) -> str:
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
    )
    return completion.choices[0].message.content

def get_10k_from_ticker(ticker: str) -> str:
    with open(f"FormData/10k_text/{ticker}.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

# -----------------------------
# 2. Strip Table of Contents
# -----------------------------
ITEM_1_BUSINESS_PATTERN = re.compile(
    r'\n\s*ITEM\s+1\.\s+BUSINESS',
    re.IGNORECASE
)

def strip_table_of_contents(text: str) -> str:
    match = ITEM_1_BUSINESS_PATTERN.search(text)
    if not match:
        raise ValueError("Could not find 'ITEM 1. BUSINESS' — check file format")
    return text[match.start():]


# -----------------------------
# 3. Split into ITEM sections
# -----------------------------
ITEM_SPLIT_PATTERN = re.compile(
    r'\n\s*(ITEM\s+\d+[A-Z]?\.)',
    re.IGNORECASE
)

def split_into_items(text: str) -> Dict[str, Dict]:
    """Split text into ITEM sections, keeping only the longest body per item."""
    parts = ITEM_SPLIT_PATTERN.split(text)

    sections: Dict[str, Dict] = {}
    for i in range(1, len(parts), 2):
        item = parts[i].strip().upper()
        body = parts[i + 1].strip()
        if item not in sections or len(body) > len(sections[item]["text"]):
            sections[item] = {"item": item, "text": body}
    return sections


# -----------------------------
# 4. Clean section text
# -----------------------------
def clean_text(text: str) -> str:
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


# -----------------------------
# 5. Drop junk / short sections
# -----------------------------
def filter_sections(sections: Dict[str, Dict], min_chars: int = 2000) -> Dict[str, Dict]:
    return {k: v for k, v in sections.items() if len(v["text"]) >= min_chars}



if __name__ == "__main__":
    ticker = "ABCL"
    text = get_10k_from_ticker(ticker)
    text = strip_table_of_contents(text)
    sections = split_into_items(text)
    for key, section in sections.items():
        section["text"] = clean_text(section["text"])
        print(f"{section['item']}  —  {len(section['text']):,} chars")
        print("-" * 100)
    sections = filter_sections(sections)
    for key, section in sections.items():
        print(key)
        print(len(section["text"]))
        if len(section["text"]) > 20000:
            print(f"Section {key} is too long, skipping")
            continue

    # print(f"\n{len(sections)} sections after filtering (>2000 chars)")
    # print("This is the business section summary:")
    total_string = "Summarize the following 10k section, just provide the summary, highlighting the products, competitive advantages, etc., no other text: " + sections["ITEM 1."]["text"][:16000]
    # summarize the business section
    # print(summarize_10k(total_string))