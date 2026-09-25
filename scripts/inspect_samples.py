import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def inspect_matches():
    gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=200)
    matched_rows = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')].head(10)
    
    s1_ids = list(matched_rows['source1_entity_id'])
    
    s1_df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=50000)
    s2_df = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=50000)
    s3_df = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=50000)
    
    s1_map = {r['entity_id']: r for _, r in s1_df.iterrows()}
    target_map = {r['entity_id']: r for _, r in s2_df.iterrows()}
    target_map.update({r['entity_id']: r for _, r in s3_df.iterrows()})
    
    count = 0
    for _, row in matched_rows.iterrows():
        s1 = row['source1_entity_id']
        targets = [t.strip() for t in str(row['matched_entity_ids']).split(',') if t.strip()]
        
        if s1 in s1_map:
            for t in targets:
                if t in target_map:
                    count += 1
                    r1 = s1_map[s1]
                    r2 = target_map[t]
                    print(f"=== Example {count}: {s1} <-> {t} ===")
                    print(f"  Name 1   : {r1['business_name']}")
                    print(f"  Name 2   : {r2['business_name']}")
                    print(f"  Addr 1   : {r1['business_address']}")
                    print(f"  Addr 2   : {r2['business_address']}")
                    print(f"  Country  : {r1['country']} vs {r2['country']}")
                    print("-" * 60)
                    if count >= 6:
                        return

if __name__ == '__main__':
    inspect_matches()
