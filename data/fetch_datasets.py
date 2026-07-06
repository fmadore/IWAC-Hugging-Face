import argparse
import json
import os
import sys

import numpy as np

# Suppress HuggingFace Hub symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Prevent datasets from importing torch (avoids DLL issues on CPU-only machines)
os.environ["USE_TORCH"] = "0"
# Disable HF usage tracking (avoids httpx SSL hangs on Windows)
os.environ["DO_NOT_TRACK"] = "1"
# Set httpx timeout to avoid indefinite hangs
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

try:
    from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID
except ImportError:  # venv without the editable install
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

console = Console()

# Available configs in the IWAC dataset
CONFIGS = ['articles', 'publications', 'documents', 'index', 'audiovisual', 'references', 'images']


def choose_dataset(cli_choice=None):
    """Resolve which repo to mirror: private full or public projection.

    ``cli_choice`` (``private``/``public``) skips the prompt. The private
    full mirror carries complete OCR / lemma columns (needs HF_TOKEN); the
    public projection has full text masked per row (OCR only for items whose
    bibo:content is public on Omeka).
    """
    choice = cli_choice
    if choice is None:
        console.print("\n[bold]Which dataset do you want to mirror locally?[/bold]")
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Dataset", style="green")
        table.add_column("Repo", style="white")
        table.add_column("Full text", style="white")
        table.add_row("1", "private (full)", PRIVATE_REPO_ID, "complete OCR + lemmas — needs HF_TOKEN")
        table.add_row("2", "public", PUBLIC_REPO_ID, "OCR masked per row (public-content items only)")
        console.print(table)
        sel = Prompt.ask("Choose dataset", choices=["1", "2"], default="1")
        choice = "private" if sel == "1" else "public"

    if choice == "public":
        return PUBLIC_REPO_ID, "public"
    return PRIVATE_REPO_ID, "private"


def main(dataset_id=PRIVATE_REPO_ID, label="private"):
    """Download all IWAC dataset configurations from Hugging Face Hub."""
    # Private mirror needs a token; public is open.
    token = (os.getenv("HF_TOKEN") or get_token()) if label == "private" else None
    if label == "private" and not token:
        console.print("[red]✗[/red] The private full mirror needs HF_TOKEN (set it in .env "
                      "or run `hf auth login`), or choose the public dataset instead.")
        return

    console.print(Panel.fit(
        "[bold cyan]IWAC Dataset Downloader[/bold cyan]\n"
        f"Fetching all configurations from [link={f'https://huggingface.co/datasets/{dataset_id}'}]{dataset_id}[/link] "
        f"([cyan]{label}[/cyan])"
        + ("\n[yellow]⚠ public projection: OCR/lemmas present only for public-content rows[/yellow]"
           if label == "public" else ""),
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
                    dataset = load_dataset(dataset_id, config_name, split="train", token=token)

                # Access the data
                console.print(f"[green]✓[/green] Dataset loaded: {len(dataset):,} rows")

                # Convert to pandas DataFrame and save as CSV
                df = dataset.to_pandas()

                # Convert array/list columns to JSON strings for CSV compatibility
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                        df[col] = df[col].apply(lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x) if isinstance(x, (list, np.ndarray)) else x)

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
    parser = argparse.ArgumentParser(description="Mirror the IWAC dataset locally as CSV files.")
    parser.add_argument(
        "--dataset",
        choices=["private", "public"],
        default=None,
        help="Which repo to mirror (default: interactive prompt). "
             "private = full mirror with complete OCR/lemmas (needs HF_TOKEN); "
             "public = projection with OCR masked per row.",
    )
    args = parser.parse_args()

    resolved_id, resolved_label = choose_dataset(args.dataset)
    main(dataset_id=resolved_id, label=resolved_label)