

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any

import requests

# -----------------------------
# ENV VARIABLES (CONFIG)
# -----------------------------
FILE_PATH: str = "unmatched_titles.csv"
API_KEY: str = "sk-72232c49afad491687b41477b4fbe874"
BASE_URL: str = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME: str = "deepseek-v4-flash"

BATCH_SIZE: int = 10
CHECKPOINT_PATH: str = "checkpoint.json"

_MAX_RETRIES: int = 5


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT: str = """
You are an AI assistant working on Ethiopian job market data.

Instructions:
- All job positions are in Ethiopian context.
- Map each input job title to the closest standardized alternative position.
- Provide a very short reason for the mapping.
- Return ONLY JSON.
- Use the index of each item as the key.

Format:
{
  "0": {"alternative_position": "...", "reason": "..."},
  "1": {"alternative_position": "...", "reason": "..."}
}
"""


# -----------------------------
# CHECKPOINT
# -----------------------------
def save_checkpoint(
    records: List[Dict[str, str]],
    next_index: int
) -> None:
    data = {
        "alternative_position": records,
        "next_index": next_index,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint() -> Dict[str, Any]:
    if not os.path.exists(CHECKPOINT_PATH):
        return {
            "alternative_position": [],
            "next_index": 0
        }

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# REQUEST WITH RETRY
# -----------------------------
def post_with_retry(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    delay = 1

    for attempt in range(1, _MAX_RETRIES + 1):
        resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=120)

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == _MAX_RETRIES:
                resp.raise_for_status()

            print(
                f"\n  [{resp.status_code}] Retrying in {delay:.0f}s ({attempt}/{_MAX_RETRIES})...",
                end="",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
            continue

        return resp.json()

    raise Exception("Max retries exceeded")


# -----------------------------
# AGENT REQUEST
# -----------------------------
def send_batch_to_agent(batch: List[str]) -> Dict[str, Any]:
    user_prompt = "Map the following job titles:\n"

    for i, title in enumerate(batch):
        user_prompt += f"{i}. {title}\n"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = post_with_retry(payload)

    content = response["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("\nFailed to parse JSON. Raw output:\n", content)
        raise


# -----------------------------
# MAIN
# -----------------------------
def process_positions() -> None:
    df = pd.read_csv(FILE_PATH)

    # Ensure columns exist
    if "alternative_position" not in df.columns:
        df["alternative_position"] = ""

    if "reason" not in df.columns:
        df["reason"] = ""

    checkpoint = load_checkpoint()

    start_index: int = checkpoint["next_index"]
    saved_records: List[Dict[str, str]] = checkpoint["alternative_position"]

    total = len(df)

    print(f"Starting from index: {start_index} / {total}")

    for i in range(start_index, total, BATCH_SIZE):
        batch_df = df.iloc[i: i + BATCH_SIZE]
        batch_titles = batch_df["title_en"].tolist()

        result = send_batch_to_agent(batch_titles)

        for j, (idx, row) in enumerate(batch_df.iterrows()):
            key = str(j)

            title = row["title_en"]
            alt_position = result.get(key, {}).get("alternative_position", "")
            reason = result.get(key, {}).get("reason", "")

            # ✅ Ensure title match before writing
            if df.at[idx, "title_en"] == title:
                df.at[idx, "alternative_position"] = alt_position
                df.at[idx, "reason"] = reason

            # Save structured checkpoint record
            saved_records.append({
                "titles_en": title,
                "alternative_position": alt_position,
                "reason": reason
            })

        next_index = i + BATCH_SIZE

        # Save checkpoint
        save_checkpoint(saved_records, next_index)

        # Save CSV progress
        df.to_csv(FILE_PATH, index=False)

        completed = min(next_index, total)
        remaining = total - completed
        print(f"  {next_index}/{total} titles processed...", end="\r", flush=True)
        time.sleep(0.5)
        # print(f"Completed: {completed} | Remaining: {remaining}")

    print("Processing completed!")



if __name__ == "__main__":
    process_positions()