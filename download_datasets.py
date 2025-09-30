import os
import json
from datasets import load_dataset

def download_and_process_datasets():
    unified = []
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    print("Downloading and processing datasets automatically...")
    
    # 1. Summarization Datasets
    print("📰 Loading summarization datasets...")
    
    # CNN/DailyMail
    try:
        cnn_dataset = load_dataset("cnn_dailymail", "3.0.0")
        for split in cnn_dataset.keys():
            for example in cnn_dataset[split]:
                article = example.get("article", "").strip()
                highlights = example.get("highlights", "").strip()
                if article and highlights:
                    unified.append({
                        "dataset": "cnn_dailymail",
                        "task": "summarization",
                        "input": article,
                        "target": highlights
                    })
        print(f"✅ CNN/DailyMail: {len([x for x in unified if x['dataset'] == 'cnn_dailymail']):,} examples")
    except Exception as e:
        print(f"❌ CNN/DailyMail error: {e}")
    
    # XSum (Extreme Summarization)
    try:
        xsum_dataset = load_dataset("xsum")
        for split in xsum_dataset.keys():
            for example in xsum_dataset[split]:
                document = example.get("document", "").strip()
                summary = example.get("summary", "").strip()
                if document and summary:
                    unified.append({
                        "dataset": "xsum",
                        "task": "summarization",
                        "input": document,
                        "target": summary
                    })
        print(f"✅ XSum: {len([x for x in unified if x['dataset'] == 'xsum']):,} examples")
    except Exception as e:
        print(f"❌ XSum error: {e}")
    
    # 2. Paraphrase Datasets
    print("\n🔄 Loading paraphrase datasets...")
    
    # PAWS (Paraphrase Adversaries from Word Scrambling)
    try:
        paws_dataset = load_dataset("paws", "labeled_final")
        for split in paws_dataset.keys():
            for example in paws_dataset[split]:
                if example.get("label") == 1:
                    s1 = str(example.get("sentence1", "")).strip()
                    s2 = str(example.get("sentence2", "")).strip()
                    if s1 and s2:
                        unified.append({
                            "dataset": "paws",
                            "task": "paraphrase",
                            "input": s1,
                            "target": s2
                        })
        print(f"✅ PAWS: {len([x for x in unified if x['dataset'] == 'paws']):,} examples")
    except Exception as e:
        print(f"❌ PAWS error: {e}")
    
    # MRPC (Microsoft Research Paraphrase Corpus)
    try:
        mrpc_dataset = load_dataset("glue", "mrpc")
        for split in mrpc_dataset.keys():
            for example in mrpc_dataset[split]:
                if example.get("label") == 1:
                    s1 = str(example.get("sentence1", "")).strip()
                    s2 = str(example.get("sentence2", "")).strip()
                    if s1 and s2:
                        unified.append({
                            "dataset": "mrpc",
                            "task": "paraphrase",
                            "input": s1,
                            "target": s2
                        })
        print(f"✅ MRPC: {len([x for x in unified if x['dataset'] == 'mrpc']):,} examples")
    except Exception as e:
        print(f"❌ MRPC error: {e}")
    
    # STSB (Semantic Textual Similarity Benchmark)
    try:
        stsb_dataset = load_dataset("stsb_multi_mt", "en")
        for split in stsb_dataset.keys():
            for example in stsb_dataset[split]:
                if example.get("similarity_score", 0) >= 4.0:
                    s1 = str(example.get("sentence1", "")).strip()
                    s2 = str(example.get("sentence2", "")).strip()
                    if s1 and s2:
                        unified.append({
                            "dataset": "stsb",
                            "task": "paraphrase",
                            "input": s1,
                            "target": s2
                        })
        print(f"✅ STSB: {len([x for x in unified if x['dataset'] == 'stsb']):,} examples")
    except Exception as e:
        print(f"❌ STSB error: {e}")
    
    # 3. Additional summarization dataset: Newsroom
    try:
        newsroom_dataset = load_dataset("newsroom")
        for split in newsroom_dataset.keys():
            for example in newsroom_dataset[split]:
                text = example.get("text", "").strip()
                summary = example.get("summary", "").strip()
                if text and summary:
                    unified.append({
                        "dataset": "newsroom",
                        "task": "summarization",
                        "input": text,
                        "target": summary
                    })
        print(f"✅ Newsroom: {len([x for x in unified if x['dataset'] == 'newsroom']):,} examples")
    except Exception as e:
        print(f"❌ Newsroom error: {e}")
    
    # Save the unified dataset
    output_file = "datasets/unified_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in unified:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n🎉 Success! Unified dataset saved to: {output_file}")
    print(f"📊 Total examples: {len(unified):,}")
    
    # Show detailed breakdown
    from collections import Counter
    dataset_counts = Counter(item['dataset'] for item in unified)
    task_counts = Counter(item['task'] for item in unified)
    
    print("\n📈 Dataset Breakdown:")
    for dataset, count in dataset_counts.items():
        print(f"   {dataset}: {count:,} examples")
    
    print("\n🎯 Task Breakdown:")
    for task, count in task_counts.items():
        print(f"   {task}: {count:,} examples")

if __name__ == "__main__":
    download_and_process_datasets()