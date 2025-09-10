import pandas as pd
from torch.utils.data import Dataset
import constants

def create_prompts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    prompts = []
    for _, row in df.iterrows():
        text = f"""
r/{row.subreddit}
Rule: {row.rule}

1) {row.positive_example_1}
Violation: Yes

2) {row.negative_example_1}
Violation: No

3) {row.negative_example_2}
Violation: No

4) {row.positive_example_2}
Violation: Yes

5) {row.body}
"""
        messages = [
            {"role": "system", "content": constants.SYS_PROMPT},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        ) + "Answer:"
        prompts.append(prompt)

    df["text"] = prompts
    df["label"] = df["rule_violation"].apply(lambda x: "Yes" if x == 1 else "No")
    df["text"] = df["text"] + df["label"]
    return df

def preprocess_df(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    items = []
    for _, row in df.iterrows():
        item = tokenizer(row["text"], add_special_tokens=False, truncation=False)
        items.append(item)
    
    processed_df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(items)
    ], axis=1)
    return processed_df

class ClassifyDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        row = self.df.iloc[index]
        return {"input_ids": row["input_ids"]}