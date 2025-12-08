from datasets import load_dataset
import os

# Suppress HuggingFace Hub symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

console = Console()

# Available configs in the IWAC dataset
CONFIGS = ['articles', 'publications', 'documents', 'index', 'audiovisual', 'references']
DATASET_ID = "fmadore/islam-west-africa-collection"


def main():
    """Download all IWAC dataset configurations from Hugging Face Hub."""
    console.print(Panel.fit(
        "[bold cyan]IWAC Dataset Downloader[/bold cyan]\n"
        f"Fetching all configurations from [link={f'https://huggingface.co/datasets/{DATASET_ID}'}]{DATASET_ID}[/link]",
        border_style="cyan"
    ))

    # Process all IWAC configs
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("[cyan]Processing configs...", total=len(CONFIGS))
        
        results = []
        
        for config_name in CONFIGS:
            progress.update(main_task, description=f"[cyan]Processing {config_name}...")
            
            console.print(f"\n[bold blue]→[/bold blue] Loading config: [cyan]{config_name}[/cyan]")
            
            try:
                # Load the dataset with specific config from the unified IWAC dataset
                with console.status(f"[bold green]Downloading {config_name} from HF Hub...", spinner="dots"):
                    dataset = load_dataset(DATASET_ID, config_name)
                
                # Access the data
                console.print(f"[green]✓[/green] Dataset loaded: {len(dataset['train']):,} rows")
                
                # Convert to pandas DataFrame and save as CSV
                df = dataset['train'].to_pandas()
                csv_filename = f"iwac_{config_name}.csv"
                csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
                
                with console.status("[bold green]Saving to CSV...", spinner="dots"):
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                
                console.print(f"[green]✓[/green] CSV saved: [cyan]{csv_filename}[/cyan] ({len(df):,} rows, {len(df.columns)} columns)")
                
                results.append({
                    'config': config_name,
                    'status': 'success',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'path': csv_path
                })
                
            except Exception as e:
                console.print(f"[red]✗[/red] Error loading {config_name}: {str(e)}")
                if config_name == 'index':
                    console.print("[yellow]ℹ[/yellow] Note: Index subset may not be available yet. Run upload_index_hf.py first.")
                
                results.append({
                    'config': config_name,
                    'status': 'failed',
                    'error': str(e)
                })
            
            progress.update(main_task, advance=1)

    # Create summary table
    console.print()
    table = Table(title="Download Summary", box=box.ROUNDED, border_style="cyan")
    table.add_column("Config", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Columns", justify="right", style="blue")
    table.add_column("File", style="dim")

    for result in results:
        if result['status'] == 'success':
            status_icon = "[green]✓[/green]"
            rows = f"{result['rows']:,}"
            cols = str(result['columns'])
            file_name = os.path.basename(result['path'])
        else:
            status_icon = "[red]✗[/red]"
            rows = "-"
            cols = "-"
            file_name = "[dim]not created[/dim]"
        
        table.add_row(result['config'], status_icon, rows, cols, file_name)

    console.print(table)

    # Final summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    total_rows = sum(r['rows'] for r in results if r['status'] == 'success')

    if failed == 0:
        console.print(Panel(
            f"[bold green]✓ All {successful} datasets downloaded successfully![/bold green]\n"
            f"Total: [cyan]{total_rows:,}[/cyan] rows across all configs",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠ {successful} succeeded, {failed} failed[/bold yellow]\n"
            f"Total: [cyan]{total_rows:,}[/cyan] rows downloaded",
            border_style="yellow"
        ))


if __name__ == "__main__":
    main()