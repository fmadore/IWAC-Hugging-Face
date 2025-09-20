from datasets import load_dataset
import os

# Available configs in the IWAC dataset
configs = ['articles', 'publications', 'documents', 'index', 'audiovisual', 'references']

# Process all IWAC configs
for config_name in configs:
    print(f"\n{'='*50}")
    print(f"Loading IWAC config: {config_name}")
    print(f"{'='*50}")
    
    try:
        # Load the dataset with specific config from the unified IWAC dataset
        dataset = load_dataset("fmadore/islam-west-africa-collection", config_name)
        
        # Access the data
        print(f"Dataset info for '{config_name}':")
        print(dataset)
        print(f"\nFirst example from '{config_name}':")
        print(dataset['train'][0])  # View first example
        
        # Convert to pandas DataFrame and save as CSV
        df = dataset['train'].to_pandas()
        csv_filename = f"iwac_{config_name}.csv"
        csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
        
        print(f"\nSaving {config_name} dataset to CSV file: {csv_path}")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV file saved successfully! ({len(df)} rows)")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error loading {config_name} from IWAC dataset: {e}")
        if config_name == 'index':
            print("Note: Index subset may not be available yet. Run upload_index_hf.py first.")

print(f"\n{'='*50}")
print("Dataset download process completed!")
print("Files created:")

# List all created files
for config_name in configs:
    csv_filename = f"iwac_{config_name}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    if os.path.exists(csv_path):
        print(f"  ✓ {csv_path}")
    else:
        print(f"  ✗ {csv_path} (not created)")

print(f"{'='*50}")