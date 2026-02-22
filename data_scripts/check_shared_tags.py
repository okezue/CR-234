import csv
import os
from datetime import datetime

# Path definitions
WORKER_0_PATH = '../data/scraped_data/battle_chunks/worker_0_results.csv'
WORKER_1_PATH = '../data/scraped_data/battle_chunks/worker_1_results.csv'
META_PATH = '../data/scraped_data/battle_meta_data.csv'
RESULTS_CSV = './tag_comparison_results.csv'


def get_tags_from_worker_files():
    """Extract tags from first column of worker_0 and worker_1 results"""
    worker_tags = set()
    
    for file_path in [WORKER_0_PATH, WORKER_1_PATH]:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist")
            continue
        
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:  # Skip empty rows
                    worker_tags.add(row[0])
    
    return worker_tags


def get_tags_from_meta():
    """Extract replayTag from battle_meta_data.csv and remove leading hashtag"""
    meta_tags = set()
    
    if not os.path.exists(META_PATH):
        print(f"Warning: {META_PATH} does not exist")
        return meta_tags
    
    with open(META_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and 'replayTag' in row:
                tag = row['replayTag'].strip()
                # Remove leading hashtag if present
                if tag.startswith('#'):
                    tag = tag[1:]
                if tag:
                    meta_tags.add(tag)
    
    return meta_tags


def main():
    timestamp = datetime.now()
    
    print("Extracting tags from worker files...")
    worker_tags = get_tags_from_worker_files()
    print(f"Worker files contain {len(worker_tags)} unique tags")
    
    print("\nExtracting tags from battle_meta_data.csv...")
    meta_tags = get_tags_from_meta()
    print(f"Meta data contains {len(meta_tags)} unique tags")
    
    # Find shared tags
    shared_tags = worker_tags & meta_tags
    print(f"\nShared tags between worker files and meta data: {len(shared_tags)}")
    
    # Save results to CSV
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'worker_tags_count', 'meta_tags_count', 'shared_tags_count'])
        writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), len(worker_tags), len(meta_tags), len(shared_tags)])
    
    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == '__main__':
    main()
