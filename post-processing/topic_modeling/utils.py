"""
utils.py
--------
Misc utilities: logging, config discovery, and interactive selection.
Uses Rich for console output following project conventions.
"""
from __future__ import annotations

import argparse
import logging
from typing import List

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich import box

# Shared console instance
console = Console()


def configure_logging() -> None:
    """Configure logging with Rich handler for beautiful output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )


def get_available_configs(repo_id: str, token: str) -> List[str]:
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(repo_id, token=token)
        if hasattr(info, 'config_names') and info.config_names:
            return list(info.config_names)
        return ['articles', 'publications']
    except Exception:
        return ['articles', 'publications']


def choose_config(available_configs: List[str]) -> str:
    """Interactive config selection with Rich UI."""
    if len(available_configs) == 1:
        console.print(f"[green]✓[/green] Une seule configuration disponible: [cyan]{available_configs[0]}[/cyan]")
        return available_configs[0]

    # Create a table for available configs
    table = Table(title="Configurations disponibles", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Configuration", style="green")
    
    for i, config in enumerate(available_configs, 1):
        table.add_row(str(i), config)
    
    console.print(table)

    while True:
        try:
            choice = IntPrompt.ask(
                f"[yellow]→[/yellow] Choisissez une configuration",
                choices=[str(i) for i in range(1, len(available_configs) + 1)],
                show_choices=False,
            )
            return available_configs[choice - 1]
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Opération annulée.")
            raise SystemExit(0)


def choose_modeling_mode() -> str:
    """Interactive mode selection with Rich UI."""
    console.print()
    panel_content = (
        "[cyan]1.[/cyan] Entraîner un nouveau modèle BERTopic [dim](recommandé)[/dim]\n"
        "[cyan]2.[/cyan] Utiliser un modèle BERTopic existant"
    )
    console.print(Panel(panel_content, title="Mode de modélisation", border_style="blue"))

    while True:
        try:
            choice = Prompt.ask(
                "[yellow]→[/yellow] Choisissez un mode",
                choices=["1", "2"],
                show_choices=False,
            )
            return "fit" if choice == "1" else "predict"
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Opération annulée.")
            raise SystemExit(0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ajoute des colonnes de modélisation de sujets à un dataset Hugging Face."
    )
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection")
    parser.add_argument(
        "--embedding-model",
        default="dangvantuan/sentence-camembert-base",
        help="Modèle d'embedding à utiliser (recommandé: sentence-camembert-base pour le français)",
    )
    parser.add_argument("--min-topic-size", type=int, default=30, help="Taille minimale des clusters (augmenté pour réduire fragmentation)")
    parser.add_argument("--model-path", default="bertopic_model")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Force l'utilisation du CPU uniquement (optimisations pour machines sans GPU)"
    )
    parser.add_argument(
        "--max-documents", type=int, default=None, help="Limite le nombre de documents pour les tests (utile pour CPU)"
    )
    parser.add_argument(
        "--embedding-batch-size", type=int, default=16, help="Taille des batches pour les embeddings (réduire si mémoire limitée)"
    )
    parser.add_argument(
        "--min-train-tokens", type=int, default=5, help="Longueur minimale (en tokens) pour inclure un texte dans l'entraînement"
    )
    parser.add_argument(
        "--domain-stopwords-file",
        type=str,
        default=None,
        help="Fichier texte (UTF-8) de stopwords supplémentaires spécifiques au domaine (1 mot par ligne)",
    )
    parser.add_argument(
        "--desired-topics",
        type=int,
        default=80,
        help="Nombre de sujets visé (défaut: 80); ajuste dynamiquement min_cluster_size = max(30, N_docs/desired_topics)",
    )
    # UMAP/HDBSCAN/Vectorizer advanced tuning
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors (BERTopic default: 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist (0.0 = tighter clusters)")
    parser.add_argument("--umap-n-components", type=int, default=5, help="UMAP n_components")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric")
    parser.add_argument(
        "--hdbscan-min-samples", type=int, default=10, help="HDBSCAN min_samples (lower = more topics, BERTopic default: 10)"
    )
    parser.add_argument(
        "--hdbscan-selection-method", type=str, choices=["eom", "leaf"], default="leaf", help="HDBSCAN cluster_selection_method (leaf = more fine-grained topics)"
    )
    parser.add_argument("--hdbscan-epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon (0.0 = no merging)")
    # Note: min_df and max_df are hardcoded in modeling.py for BERTopic c-TF-IDF compatibility
    # min_df=2, max_df=1.0 - these cannot be changed without risking sklearn errors
    parser.add_argument("--vectorizer-max-features", type=int, default=25000, help="CountVectorizer max_features")
    parser.add_argument("--vectorizer-ngrams", type=str, default="1,3", help="CountVectorizer ngram range 'a,b'")
    # Outlier reduction options
    parser.add_argument(
        "--reduce-outliers-train",
        type=float,
        default=0.3,
        help="Seuil (0-1) pour réduire/réassigner les outliers à l'entraînement via c-TF-IDF (défaut: 0.3)",
    )
    parser.add_argument(
        "--outlier-reassign-threshold",
        type=float,
        default=0.2,
        help="Réassigne un outlier à la prédiction si la meilleure proba >= seuil (défaut: 0.2)",
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Réduire le nombre final de topics après entraînement (merge similar topics). Ex: --nr-topics 60",
    )
    parser.add_argument(
        "--topic-label-max-words", type=int, default=6, help="Nombre maximum de mots uniques dans les labels de topics (défaut: 6)"
    )
    
    # Digital Humanities quality metrics (enabled by default)
    parser.add_argument(
        "--skip-coherence",
        action="store_true",
        help="Ne pas calculer les métriques de cohérence (activé par défaut si gensim disponible).",
    )
    
    # Topic-over-time analysis (enabled by default if pub_date exists)
    parser.add_argument(
        "--skip-topics-over-time",
        action="store_true",
        help="Ne pas calculer l'évolution temporelle des topics (activé par défaut si pub_date existe).",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=None,
        help="Nombre de bins temporels pour l'agrégation topics_over_time (défaut: automatique par BERTopic).",
    )
    parser.add_argument(
        "--save-topics-over-time",
        type=str,
        default=None,
        help="Chemin pour sauvegarder le DataFrame topics_over_time en CSV (pour visualisation externe).",
    )
    
    return parser


def display_coherence_summary(metrics: dict) -> None:
    """Display coherence metrics in a Rich table."""
    if "error" in metrics:
        console.print(f"[yellow]⚠[/yellow] Cohérence non calculée: {metrics['error']}")
        return
    
    table = Table(title="Métriques de Cohérence des Topics", box=box.ROUNDED)
    table.add_column("Métrique", style="cyan")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Interprétation", style="dim")
    
    for metric_name in ["c_v", "npmi", "u_mass", "topic_diversity"]:
        if metric_name in metrics and "score" in metrics[metric_name]:
            data = metrics[metric_name]
            score = f"{data['score']:.4f}"
            interpretation = data.get("interpretation", "")
            table.add_row(metric_name.upper(), score, interpretation)
    
    console.print(table)
    
    # Quality assessment
    if "c_v" in metrics and "score" in metrics["c_v"]:
        cv = metrics["c_v"]["score"]
        if cv >= 0.5:
            console.print("[green]✓[/green] Bonne cohérence (C_v ≥ 0.5) - Topics interprétables")
        elif cv >= 0.4:
            console.print("[yellow]ℹ[/yellow] Cohérence acceptable (C_v 0.4-0.5) - Topics utilisables")
        else:
            console.print("[yellow]⚠[/yellow] Cohérence faible (C_v < 0.4) - Considérez ajuster les paramètres")
