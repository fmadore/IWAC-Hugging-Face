from datasets import load_dataset
import os

# Available configs in the dataset
configs = ['articles', 'publications']

for config_name in configs:
    print(f"\n{'='*50}")
    print(f"Loading config: {config_name}")
    print(f"{'='*50}")
    
    # Load the dataset with specific config
    dataset = load_dataset("fmadore/iwac-newspaper-articles", config_name)
    
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

print(f"\n{'='*50}")
print("All datasets downloaded successfully!")
print("Files created:")
for config_name in configs:
    csv_filename = f"iwac_{config_name}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    print(f"  - {csv_path}")
print(f"{'='*50}")