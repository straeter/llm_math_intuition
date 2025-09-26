import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.io_utils import load_jsonl

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# st.html("""<style>[alt=Logo] {height: 3rem;}</style>""")
# st.logo("static/llm_values.jpg")

def highlight_correct(row):
    if row["Correct"] == "True":
        return ["background-color: lightgreen"] * len(row)
    return [""] * len(row)


@st.cache_data
def load_data():
    models = []
    for file in os.listdir("data"):
        if file.startswith("responses_") and file.endswith("_verified.jsonl"):
            models.append(file.split("_")[1])
    models = sorted(list(set(models)))

    data = []
    for model in models:
        lines = load_jsonl(f"data/responses_{model}_verified.jsonl")
        data += lines

    categories = {}
    for line in data:
        cat, subcat = line["module"].split("__")

        if cat not in categories:
            categories[cat] = set()
        categories[cat].add(subcat)
    categories = dict(sorted(categories.items()))
    for cat in categories.keys():
        categories[cat] = sorted(list(categories[cat]))

    return data, models, categories


def process_data_by_model(data):
    # stats[model][category][subcategory] = list of correct booleans
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for entry in data:
        model = entry["model"]
        module = entry["module"]
        correct = entry["correct"]

        if "__" in module:
            category, subcategory = module.split("__", 1)
        else:
            category, subcategory = module, "misc"

        stats[model][category][subcategory].append(correct)

    return stats


def compute_all_ratios(stats_by_model):
    # ratios[model][category][subcategory] = accuracy
    all_ratios = defaultdict(lambda: defaultdict(dict))

    for model, categories in stats_by_model.items():
        for category, subcats in categories.items():
            for subcat, results in subcats.items():
                if results:
                    all_ratios[model][category][subcat] = sum(results) / len(results)
                else:
                    all_ratios[model][category][subcat] = 0.0
    return all_ratios


def plot_all_models(all_ratios, models, category):
    """Grouped bar plot for a single category (chosen from Streamlit sidebar)."""
    subcats = sorted({sub for m in models if category in all_ratios[m]
                      for sub in all_ratios[m][category]})
    x = np.arange(len(subcats))  # positions for subcategories

    bar_width = 0.5 / len(models)  # divide space between models

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, model in enumerate(models):
        ratios = all_ratios[model]
        if category not in ratios:
            continue

        y = [ratios[category].get(sub, 0) for sub in subcats]

        # shift each model's bars sideways
        ax.bar(x + idx * bar_width, y, width=bar_width, label=model)

    ax.set_title(f"Category: {category}")
    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(subcats, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    return fig


def main():
    # --- Streamlit app ---
    st.title("Model Accuracy Explorer")
    data, models, categories = load_data()
    selected_models = st.multiselect(
        "Models are prompted to answer without CoT, thinking, reasoning etc. - we also adjusted all parameters to prevent any form of hidden reasoning.",
        models, default=models)

    with st.sidebar:
        st.markdown(
            "<div style='font-weight: 600'>Explore mathematical intuition for LLMs</div>",
            unsafe_allow_html=True
        )
        category = st.selectbox("Choose a category:", categories, index=0, key="topic")
        # subcategories = list(set([q["module"].split("__")[1] for q in data if q["module"].startswith(category)]))
        subcategories = categories[category]
        subcategory = st.selectbox("Choose a subcategory:", subcategories, index=0, key="subcategory")
        questions_shown = [q for q in data if
                           q["model"] in selected_models
                           and q["module"].split("__")[0] == category
                           and q["module"].split("__")[1] == subcategory]

        q_table = pd.DataFrame(
            {
                "Category": [q["module"] for q in questions_shown],
                "Question": [q["question"] for q in questions_shown],
                "LLM answer": [q["response"] for q in questions_shown],
                "Correct answer": [q["answer"] for q in questions_shown],
                "Model": [str(q["model"]) for q in questions_shown],
                "Correct": [str(q["correct"]) for q in questions_shown]
            },
            # index=["Actual Cat", "Actual Dog", "Actual Bird", "Actual Fish"],
        )
        styled_df = q_table.style.apply(highlight_correct, axis=1)

    stats_by_model = process_data_by_model(data)
    all_ratios = compute_all_ratios(stats_by_model)
    fig = plot_all_models(all_ratios, selected_models, category)
    st.pyplot(fig, width=1000)

    st.dataframe(styled_df, width="content")


if __name__ == '__main__':
    main()
