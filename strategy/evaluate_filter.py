from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import shorten

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.strategy_rag import (
    CHUNK_SIZE,
    COLLECTION_NAME,
    METADATA_PATH,
    PROCESSED_DIR,
    build_store,
    infer_category_filter,
    load_and_chunk,
    load_metadata,
    make_qwen_embedder,
)

TEST_QUERIES = [
    {
        "question": "Bệnh Alzheimer có di truyền không?",
        "expected_category": "Thần kinh",
    },
    {
        "question": "Nguyên nhân gây ăn không tiêu là gì?",
        "expected_category": "Tiêu hóa - Gan mật",
    },
    {
        "question": "Áp xe hậu môn có tự khỏi không?",
        "expected_category": "Tiêu hóa - Hậu môn trực tràng",
    },
    {
        "question": "Triệu chứng của áp xe phổi?",
        "expected_category": "Hô hấp",
    },
    {
        "question": "Bàn chân đái tháo đường chăm sóc thế nào?",
        "expected_category": "Nội tiết - Đái tháo đường",
    },
    {
        "question": "Băng huyết sau sinh xử lý ra sao?",
        "expected_category": "Sản phụ khoa",
    },
    {
        "question": "Bàng quang tăng hoạt là gì?",
        "expected_category": "Tiết niệu",
    },
]


def _fmt(text: str, width: int = 60) -> str:
    return shorten(text.replace("\n", " "), width=width, placeholder="…")


def evaluate(store, meta_map: dict, top_k: int = 3):

    print()
    print("=" * 100)
    print("  METADATA FILTER EVALUATION: search() vs search_with_filter()")
    print("=" * 100)

    total_queries = len(TEST_QUERIES)
    summary = {
        "filter_precision_sum": 0.0,
        "no_filter_precision_sum": 0.0,
        "overlap_sum": 0,
        "filter_empty_count": 0,
    }

    for qi, tq in enumerate(TEST_QUERIES, 1):
        question = tq["question"]
        expected_cat = tq["expected_category"]

        auto_filter = infer_category_filter(question, meta_map)

        results_no_filter = store.search(question, top_k=top_k)

        if auto_filter:
            results_filtered = store.search_with_filter(
                question, top_k=top_k, metadata_filter=auto_filter
            )
        else:
            results_filtered = results_no_filter  # fallback

        def precision(results, expected):
            if not results:
                return 0.0
            hits = sum(
                1 for r in results
                if r.get("metadata", {}).get("category", "") == expected
            )
            return hits / len(results)

        prec_no_filter = precision(results_no_filter, expected_cat)
        prec_filtered = precision(results_filtered, expected_cat)

        ids_no_filter = {r["id"] for r in results_no_filter}
        ids_filtered = {r["id"] for r in results_filtered}
        overlap = len(ids_no_filter & ids_filtered)

        if not results_filtered:
            summary["filter_empty_count"] += 1

        summary["no_filter_precision_sum"] += prec_no_filter
        summary["filter_precision_sum"] += prec_filtered
        summary["overlap_sum"] += overlap

        detected = auto_filter.get("category", "—") if auto_filter else "None"
        print(f"\n{'─' * 100}")
        print(f"  Q{qi}: {question}")
        print(f"  Expected category: {expected_cat}")
        print(f"  Auto-detected filter: {detected}")
        print()

        print(f"  {'Rank':<5} {'search() — no filter':<47} {'search_with_filter()':<47}")
        print(f"  {'─'*5} {'─'*47} {'─'*47}")

        max_rows = max(len(results_no_filter), len(results_filtered))
        for i in range(max_rows):
            if i < len(results_no_filter):
                r = results_no_filter[i]
                cat = r.get("metadata", {}).get("category", "")
                score = r.get("score", 0)
                match = "v" if cat == expected_cat else "v"
                col1 = f"{match} {cat[:20]:<20s} score={score:.4f}"
            else:
                col1 = ""

            if i < len(results_filtered):
                r = results_filtered[i]
                cat = r.get("metadata", {}).get("category", "")
                score = r.get("score", 0)
                match = "v" if cat == expected_cat else "x"
                col2 = f"{match} {cat[:20]:<20s} score={score:.4f}"
            else:
                col2 = ""

            print(f"  [{i+1}]   {col1:<47} {col2:<47}")

        print()
        print(f"  Precision (no filter): {prec_no_filter:.0%}   |   "
              f"Precision (filtered): {prec_filtered:.0%}   |   "
              f"Overlap: {overlap}/{top_k}")

    print()

    avg_prec_no = summary["no_filter_precision_sum"] / total_queries
    avg_prec_fi = summary["filter_precision_sum"] / total_queries
    avg_overlap = summary["overlap_sum"] / total_queries

    print(f"  Total queries evaluated:         {total_queries}")
    print(f"  Avg precision (no filter):       {avg_prec_no:.1%}")
    print(f"  Avg precision (with filter):     {avg_prec_fi:.1%}")
    print(f"  Precision improvement:           {avg_prec_fi - avg_prec_no:+.1%}")
    print(f"  Avg result overlap:              {avg_overlap:.1f} / {top_k}")
    print(f"  Queries where filter was empty:  {summary['filter_empty_count']}")
    print()

    # Verdict
    if avg_prec_fi > avg_prec_no:
        print("Filter Effectiveness: Filtering by category IMPROVES precision.")
    elif avg_prec_fi == avg_prec_no:
        print("Filter Effectiveness: No difference — results already well-targeted.")
    else:
        print("Filter Effectiveness: Filtering HURT precision (possible keyword mismatch).")

    if summary["filter_empty_count"] > 0:
        print(f"Recall Trade-off: {summary['filter_empty_count']} queries returned "
              "EMPTY results with filter (too restrictive).")
    else:
        print(" Trade-off: No queries lost results due to filtering.")

    if avg_overlap < top_k * 0.5:
        print("Low overlap — filter is selecting substantially different chunks.")
    else:
        print("High overlap — filtered and unfiltered results are mostly the same.")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate metadata filter utility")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    meta_map = load_metadata()
    embed_fn = make_qwen_embedder()

    print("Loading & chunking ...")
    documents = load_and_chunk(PROCESSED_DIR, chunk_size=args.chunk_size)
    print(f"Total chunks: {len(documents)}")

    print("Building vector store ...")
    store = build_store(documents, embed_fn, collection_name="eval_filter_test")

    evaluate(store, meta_map, top_k=args.top_k)


if __name__ == "__main__":
    main()
