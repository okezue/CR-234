"""
DOES NOT WORK DO NOT USE
"""

import os
import csv
from datetime import datetime

SCRAPED_DATA_DIR = '../data/scraped_data'
CHECKPOINT_BASE_DIR = '../data/check_point_data'


def get_latest_checkpoint_folder(base_dir):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        return None
    folders.sort(reverse=True)
    return os.path.join(base_dir, folders[0])


def append_csv_files(output_path, input_dirs, file_pattern=None, subdirs=None):
    header_written = False
    header_length = None
    skipped_rows = 0
    with open(output_path, 'w', newline='') as outfile:
        writer = None
        for dir_path in input_dirs:
            if not os.path.exists(dir_path):
                continue
            
            # If subdirs is specified, look in those subdirectories
            if subdirs:
                search_dirs = [os.path.join(dir_path, subdir) for subdir in subdirs if os.path.exists(os.path.join(dir_path, subdir))]
            else:
                search_dirs = [dir_path]
            
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                for file in os.listdir(search_dir):
                    if not file.endswith('.csv'):
                        continue
                    if file_pattern and file_pattern not in file:
                        continue
                    file_path = os.path.join(search_dir, file)
                    with open(file_path, 'r', newline='') as infile:
                        reader = csv.reader(infile)
                        header = next(reader)
                        if not header_written:
                            writer = csv.writer(outfile)
                            writer.writerow(header)
                            header_length = len(header)
                            header_written = True
                        for row in reader:
                            # Skip rows with mismatched field count
                            if len(row) != header_length:
                                skipped_rows += 1
                                continue
                            writer.writerow(row)
    
    if skipped_rows > 0:
        print(f'Warning: Skipped {skipped_rows} malformed rows in {output_path}')


def make_new_checkpoint():
    latest_checkpoint = get_latest_checkpoint_folder(CHECKPOINT_BASE_DIR)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, now_str)
    os.makedirs(new_checkpoint_dir, exist_ok=True)

    input_dirs = []
    if latest_checkpoint:
        input_dirs.append(latest_checkpoint)
    input_dirs.append(SCRAPED_DATA_DIR)

    # Combine metadata separately
    metadata_path = os.path.join(new_checkpoint_dir, 'all_battle_meta_data.csv')
    append_csv_files(metadata_path, input_dirs, file_pattern='all_battle_meta_data')
    
    # Combine worker data separately (from battle_chunks subdirectory)
    worker_path = os.path.join(new_checkpoint_dir, 'all_worker_rows.csv')
    append_csv_files(worker_path, input_dirs, file_pattern='worker', subdirs=['battle_chunks'])
    
    print(f'New checkpoint created at {new_checkpoint_dir}')


if __name__ == '__main__':
    make_new_checkpoint()
