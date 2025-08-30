#!/usr/bin/env python3
"""
calculate_sentiment_AI.py
===========================

Ajoute des colonnes avec l'analyse de sentiment détaillée via l'API Gemini ou ChatGPT
à un dataset Hugging Face existant. Ce script se base sur une colonne texte
spécifiée (par défaut 'OCR').

Le script :
1. Charge un dataset Hugging Face.
2. Pour chaque texte dans la colonne spécifiée :
    a. Vérifie si une analyse existe dans un cache local.
    b. Si non, appelle l'API Gemini ou ChatGPT avec un prompt structuré.
    c. Sauvegarde le résultat de l'API dans le cache.
3. Ajoute les résultats de l'analyse dans de nouvelles colonnes
   (préfixées par "gemini_" ou "chatgpt_").
4. Réorganise les colonnes pour placer les nouvelles colonnes après
   la colonne 'sentiment_score' (supposée exister).
5. Pousse le dataset modifié vers le Hugging Face Hub.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications'),
le modèle d'IA à utiliser ('gemini' ou 'chatgpt'), et peut spécifier le nom du repo,
la colonne texte, etc., via des arguments CLI.

Usage
-----
    python post-processing/calculate_sentiment_AI.py [--repo MON_USER/MON_DATASET] [--config-name CONFIG] [--text-column TEXT_COL] [--model MODEL]

Exemple:
    python post-processing/calculate_sentiment_AI.py --repo fmadore/islam-west-africa-collection --config-name articles --model gemini

Variables d'environnement
-------------------------
GOOGLE_API_KEY   Clé API pour Google Gemini.
CHATGPT          Clé API pour OpenAI ChatGPT.
HF_TOKEN         Jeton d'accès personnel pour le Hugging Face Hub.

Dépendances supplémentaires
-------------------------
    pip install datasets huggingface_hub google-api-python-client python-dotenv tqdm google-ai-generativelanguage openai
    (Note: google.genai is part of google-ai-generativelanguage or a similar package, ensure correct installation for the older SDK)
"""
import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, login
from dotenv import load_dotenv
from tqdm import tqdm

# Gemini (Google GenAI SDK - older version)
try:
    from google import genai
    from google.genai import types
    from google.genai import errors
except ImportError:
    print("Veuillez installer la librairie google-ai-generativelanguage (qui inclut google.genai): pip install google-ai-generativelanguage")
    print("Ou assurez-vous que le SDK google.genai est correctement installé.")
    genai = None

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("Veuillez installer la librairie OpenAI: pip install openai")
    OpenAI = None

# Configuration du logging
def configure_logging() -> logging.Logger:
    """Configure le logging de base."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Schema pour la validation des données (sans Pydantic pour éviter les problèmes de pickling)
EXPECTED_FIELDS = {
    "centralite_islam_musulmans": str,
    "centralite_justification": str,
    "subjectivite_score": (int, type(None)),
    "subjectivite_justification": str,
    "polarite": str,
    "polarite_justification": str
}

def validate_sentiment_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide la sortie de l'analyse de sentiment sans utiliser Pydantic."""
    validated = {}
    for field, expected_type in EXPECTED_FIELDS.items():
        value = data.get(field)
        if isinstance(expected_type, tuple):  # pour Optional[int]
            if value is not None and not isinstance(value, expected_type[0]):
                raise ValueError(f"Field {field} must be {expected_type[0]} or None, got {type(value)}")
        else:
            if not isinstance(value, expected_type):
                raise ValueError(f"Field {field} must be {expected_type}, got {type(value)}")
        validated[field] = value
    return validated

# --- Model Configuration ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
# Updated to use the newer ChatGPT 5 mini model
CHATGPT_MODEL_NAME = "gpt-5-mini"

# --- Cache Configuration ---
GEMINI_CACHE_FILE_DEFAULT_NAME = "gemini_sentiment_cache.json"
CHATGPT_CACHE_FILE_DEFAULT_NAME = "chatgpt_sentiment_cache.json"

def load_cache(cache_file_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Charge le cache depuis un fichier JSON."""
    if cache_file_path.exists():
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                logger.info(f"Chargement du cache depuis {cache_file_path}")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Impossible de charger le cache depuis {cache_file_path}: {e}. Un nouveau cache sera créé.")
    return {}

def save_cache(cache_file_path: Path, cache_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Sauvegarde le cache dans un fichier JSON."""
    try:
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Cache sauvegardé dans {cache_file_path}")
    except IOError as e:
        logger.error(f"Erreur lors de la sauvegarde du cache dans {cache_file_path}: {e}")

# Prompt pour l'analyse de sentiment (utilisé pour les deux modèles)
PROMPT_MD_RELATIVE_PATH = os.path.join("prompts", "sentiment_prompt.md")

def _load_base_prompt() -> str:
    """Load the reusable base prompt instructions from the markdown file once."""
    try:
        base_dir = Path(__file__).resolve().parent
        prompt_path = base_dir / PROMPT_MD_RELATIVE_PATH
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERREUR: impossible de charger le fichier de prompt: {e}]"

_BASE_PROMPT_MD = _load_base_prompt()

def create_sentiment_prompt(article_text: str) -> str:
    """Compose the final prompt by embedding the article text after the base instructions."""
    return f"{_BASE_PROMPT_MD}\n\nTexte à analyser:\n---\n{article_text}\n---\n\nRetournez uniquement le JSON demandé."

def analyze_text_with_gemini(
    article_text: str,
    google_api_key: str,
    model_name: str,
    logger: logging.Logger,
    max_retries: int = 3,
    initial_backoff: int = 5
) -> Dict[str, Any]:
    """
    Analyse le sentiment d'un texte d'article en utilisant l'API Gemini (google.genai SDK).
    Retourne un dictionnaire avec les champs de SentimentAnalysisOutput et un champ 'analysis_error'.
    """
    default_error_result = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": "Erreur lors de l'analyse Gemini.",
        "subjectivite_score": None,
        "subjectivite_justification": "Erreur lors de l'analyse Gemini.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": "Erreur lors de l'analyse Gemini.",
        "analysis_error": "Erreur inconnue"
    }

    if not article_text or not article_text.strip():
        logger.warning("Texte de l'article vide ou manquant pour l'analyse Gemini.")
        return {
            **default_error_result,
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte de l'article non fourni ou vide.",
            "subjectivite_justification": "Non applicable car le texte de l'article est vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable car le texte de l'article est vide.",
            "analysis_error": "Texte vide fourni pour analyse"
        }

    try:
        client = genai.Client(api_key=google_api_key)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Gemini: {e}")
        return {**default_error_result, "analysis_error": f"Erreur client Gemini: {e}"}

    prompt_content = create_sentiment_prompt(article_text)
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_content)]
        )
    ]

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config
            )

            if not response.text:
                logger.warning(f"Réponse vide de Gemini pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": "Réponse vide de Gemini après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                json_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON de la réponse Gemini (essai {attempt + 1}/{max_retries}): {e}. Réponse: {response.text[:500]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"JSONDecodeError: {e}", "raw_response_snippet": response.text[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                validated_output = validate_sentiment_output(json_response)
                return {**validated_output, "analysis_error": None}
            except (ValueError, KeyError) as e:
                logger.error(f"Erreur de validation (essai {attempt + 1}/{max_retries}): {e}. Données reçues: {json_response}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"ValidationError: {e}", "parsed_data": json_response}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

        except errors.APIError as e:
            error_code_str = f" (Code: {e.code})" if hasattr(e, 'code') else ""
            error_message_str = f" (Message: {e.message})" if hasattr(e, 'message') else ""
            logger.error(f"Erreur API Gemini (essai {attempt + 1}/{max_retries}): {e}{error_code_str}{error_message_str}. Texte (début): {article_text[:200]}")
            
            if attempt == max_retries - 1:
                return {**default_error_result, "analysis_error": f"APIError: {e}{error_code_str}{error_message_str}"}
            time.sleep(initial_backoff * (2 ** attempt) * 3)
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'appel à Gemini (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "analysis_error": f"Exception: {e}"}
            time.sleep(initial_backoff * (2 ** attempt))
            
    return {**default_error_result, "analysis_error": "Échec de l'analyse Gemini après plusieurs tentatives."}

def analyze_text_with_chatgpt(
    article_text: str,
    chatgpt_api_key: str,
    model_name: str,
    logger: logging.Logger,
    max_retries: int = 3,
    initial_backoff: int = 5
) -> Dict[str, Any]:
    """
    Analyse le sentiment d'un texte d'article en utilisant l'API ChatGPT (OpenAI gpt-5-mini).
    Retourne un dictionnaire avec les champs de SentimentAnalysisOutput et un champ 'analysis_error'.
    """
    default_error_result = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": "Erreur lors de l'analyse ChatGPT.",
        "subjectivite_score": None,
        "subjectivite_justification": "Erreur lors de l'analyse ChatGPT.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": "Erreur lors de l'analyse ChatGPT.",
        "analysis_error": "Erreur inconnue"
    }

    if not article_text or not article_text.strip():
        logger.warning("Texte de l'article vide ou manquant pour l'analyse ChatGPT.")
        return {
            **default_error_result,
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte de l'article non fourni ou vide.",
            "subjectivite_justification": "Non applicable car le texte de l'article est vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable car le texte de l'article est vide.",
            "analysis_error": "Texte vide fourni pour analyse"
        }

    try:
        client = OpenAI(api_key=chatgpt_api_key)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client ChatGPT: {e}")
        return {**default_error_result, "analysis_error": f"Erreur client ChatGPT: {e}"}

    prompt_content = create_sentiment_prompt(article_text)

    # Nous utilisons maintenant l'API Responses (client.responses.create) avec la configuration demandée.
    system_preamble = (
        "Vous êtes un expert en analyse de sentiments. "
        "Analysez le texte fourni et retournez UNIQUEMENT un objet JSON respectant strictement le schéma : "
        "{centralite_islam_musulmans:str, centralite_justification:str, subjectivite_score:int|nul, "
        "subjectivite_justification:str, polarite:str, polarite_justification:str}. "
        "Aucun texte hors JSON."
    )

    def _extract_json(text: str) -> Dict[str, Any]:
        """Tente de parser directement, sinon isole la première/dernière accolade."""
        if not text:
            raise ValueError("Réponse vide")
        try:
            return json.loads(text)
        except Exception:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                return json.loads(candidate)
            raise

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": system_preamble},
                    {"role": "user", "content": prompt_content}
                ],
                text={
                    "format": {"type": "text"},  # Conserver la config fournie
                    "verbosity": "low"
                },
                reasoning={"effort": "low"},
                tools=[],
                store=True
            )

            # Tentative d'accès au texte de sortie (selon le SDK openai >= 1.0 responses)
            raw_text = None
            if hasattr(response, 'output_text') and response.output_text:
                raw_text = response.output_text
            else:
                # Fallback : concaténer les segments
                try:
                    parts = []
                    for item in getattr(response, 'output', []) or []:
                        content = getattr(item, 'content', None)
                        if content:
                            for c in content:
                                t = getattr(c, 'text', None)
                                if t and getattr(t, 'value', None):
                                    parts.append(t.value)
                    if parts:
                        raw_text = "\n".join(parts)
                except Exception:
                    pass

            if not raw_text:
                logger.warning(f"Réponse vide de ChatGPT pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": "Réponse vide de ChatGPT après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

            try:
                json_payload = _extract_json(raw_text)
            except Exception as je:
                logger.error(f"Échec parsing JSON (essai {attempt + 1}/{max_retries}): {je}. Extrait: {raw_text[:200]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"JSON parse error: {je}", "raw_response_snippet": raw_text[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

            try:
                validated_output = validate_sentiment_output(json_payload)
            except (ValueError, KeyError) as ve:
                logger.error(f"Validation échouée (essai {attempt + 1}/{max_retries}): {ve}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"ValidationError: {ve}", "parsed_data": json_payload}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

            return {**validated_output, "analysis_error": None}

        except Exception as e:
            logger.error(f"Erreur lors de l'appel à ChatGPT (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "analysis_error": f"Exception: {e}"}
            time.sleep(initial_backoff * (2 ** attempt))

    return {**default_error_result, "analysis_error": "Échec de l'analyse ChatGPT après plusieurs tentatives."}

def process_batch_with_analysis(
    batch: Dict[str, List[Any]], 
    model_choice: str,
    api_key: str,
    model_name: str,
    cache: Dict[str, Any], 
    cache_file_path: Path, 
    logger: logging.Logger,
    text_column_name: str,
    id_column_name: str = "o:id"
) -> Dict[str, List[Any]]:
    """
    Applique l'analyse de sentiment à un batch d'exemples, utilisant et mettant à jour un cache.
    Utilise la colonne spécifiée par id_column_name comme clé de cache.
    """
    ocr_texts = batch[text_column_name]
    ids = batch[id_column_name]
    
    # Initialiser les listes pour les nouvelles colonnes
    results_centralite_islam_musulmans = []
    results_centralite_justification = []
    results_subjectivite_score = []
    results_subjectivite_justification = []
    results_polarite = []
    results_polarite_justification = []

    processed_in_batch = 0
    for record_id, text in zip(ids, ocr_texts):
        cache_key = str(record_id)

        # Vérifier si l'entrée existe dans le cache ET n'a pas d'erreur
        if cache_key in cache:
            cached_result = cache[cache_key]
            # Ignorer le cache si l'entrée contient une erreur d'analyse
            has_error = (cached_result.get("centralite_islam_musulmans") == "ERREUR_ANALYSE" or
                        cached_result.get("polarite") == "ERREUR_ANALYSE" or
                        cached_result.get("analysis_error") is not None)
            
            if has_error:
                logger.info(f"Entrée avec erreur trouvée dans le cache pour l'ID '{cache_key}', ré-analyse en cours...")
                if model_choice == "gemini":
                    analysis_result = analyze_text_with_gemini(text, api_key, model_name, logger)
                else:  # chatgpt
                    analysis_result = analyze_text_with_chatgpt(text, api_key, model_name, logger)
                cache[cache_key] = analysis_result
                processed_in_batch += 1
            else:
                analysis_result = cached_result
                logger.debug(f"Résultat trouvé dans le cache pour l'ID '{cache_key}'.")
        else:
            logger.debug(f"Analyse {model_choice.upper()} pour l'ID '{cache_key}' (texte début): {text[:50]}...")
            if model_choice == "gemini":
                analysis_result = analyze_text_with_gemini(text, api_key, model_name, logger)
            else:  # chatgpt
                analysis_result = analyze_text_with_chatgpt(text, api_key, model_name, logger)
            cache[cache_key] = analysis_result
            processed_in_batch += 1
        
        results_centralite_islam_musulmans.append(analysis_result.get("centralite_islam_musulmans"))
        results_centralite_justification.append(analysis_result.get("centralite_justification"))
        results_subjectivite_score.append(analysis_result.get("subjectivite_score"))
        results_subjectivite_justification.append(analysis_result.get("subjectivite_justification"))
        results_polarite.append(analysis_result.get("polarite"))
        results_polarite_justification.append(analysis_result.get("polarite_justification"))

    if processed_in_batch > 0:
        save_cache(cache_file_path, cache, logger)
        logger.info(f"{processed_in_batch} nouveaux éléments traités et ajoutés au cache dans ce batch.")

    # Ajouter les colonnes avec le préfixe approprié
    prefix = model_choice
    batch[f"{prefix}_centralite_islam_musulmans"] = results_centralite_islam_musulmans
    batch[f"{prefix}_centralite_justification"] = results_centralite_justification
    batch[f"{prefix}_subjectivite_score"] = results_subjectivite_score
    batch[f"{prefix}_subjectivite_justification"] = results_subjectivite_justification
    batch[f"{prefix}_polarite"] = results_polarite
    batch[f"{prefix}_polarite_justification"] = results_polarite_justification
    
    return batch

def main():
    logger = configure_logging()
    
    # Déterminer le répertoire du script et la racine du projet
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Charger les variables d'environnement depuis .env à la racine du projet
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Variables d'environnement chargées depuis {dotenv_path}")
    else:
        logger.warning(f"Fichier .env non trouvé à {dotenv_path}. Assurez-vous que les clés API sont définies.")

    parser = argparse.ArgumentParser(description="Ajoute des colonnes d'analyse de sentiment via Gemini ou ChatGPT à un dataset Hugging Face.")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset).")
    parser.add_argument("--config-name", type=str, default=None, help="Nom de la configuration à traiter (ex: 'articles', 'publications'). Sera demandé si non fourni.")
    parser.add_argument("--text-column", default="OCR", help="Nom de la colonne contenant le texte à analyser.")
    parser.add_argument("--id-column", default="o:id", help="Nom de la colonne contenant les identifiants uniques pour le cache.")
    parser.add_argument("--model", type=str, default=None, help="Modèle à utiliser ('gemini' ou 'chatgpt'). Sera demandé si non fourni.")
    parser.add_argument("--batch-size", type=int, default=10, help="Taille des batchs pour le traitement .map().")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille maximale des shards Parquet lors du push vers le Hub.")
    
    args = parser.parse_args()

    repo_id = args.repo
    text_column_name = args.text_column
    id_column_name = args.id_column
    batch_size = args.batch_size
    max_shard_size = args.max_shard_size

    # --- Choix du modèle par l'utilisateur ---
    model_choice = args.model
    if not model_choice:
        while model_choice not in ["gemini", "chatgpt"]:
            try:
                model_choice = input("Entrez le modèle à utiliser ('gemini' ou 'chatgpt'): ").strip().lower()
            except KeyboardInterrupt:
                logger.info("\nOpération annulée par l'utilisateur.")
                return
            except EOFError:
                logger.error("\nEntrée non attendue. Arrêt.")
                return
    logger.info(f"Modèle sélectionné: {model_choice}")

    # --- Configuration selon le modèle choisi ---
    if model_choice == "gemini":
        if genai is None:
            logger.error("Le SDK Google GenAI n'est pas installé. Veuillez installer: pip install google-ai-generativelanguage")
            return
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")
            return
        
        model_name = GEMINI_MODEL_NAME
        cache_file_path = Path(script_dir / GEMINI_CACHE_FILE_DEFAULT_NAME)
        
        # Test du client
        try:
            _ = genai.Client(api_key=api_key)
            logger.info(f"Clé API Google chargée. Prêt à utiliser le modèle '{model_name}'.")
        except Exception as e:
            logger.error(f"Erreur lors de la pré-vérification du client Gemini: {e}")
            return
            
    else:  # chatgpt
        if OpenAI is None:
            logger.error("Le SDK OpenAI n'est pas installé. Veuillez installer: pip install openai")
            return
        
        api_key = os.getenv("CHATGPT")
        if not api_key:
            logger.error("La variable d'environnement CHATGPT n'est pas définie.")
            return
        
        model_name = CHATGPT_MODEL_NAME
        cache_file_path = Path(script_dir / CHATGPT_CACHE_FILE_DEFAULT_NAME)
        
        # Test du client
        try:
            _ = OpenAI(api_key=api_key)
            logger.info(f"Clé API ChatGPT chargée. Prêt à utiliser le modèle '{model_name}'.")
        except Exception as e:
            logger.error(f"Erreur lors de la pré-vérification du client ChatGPT: {e}")
            return

    # --- Choix de la configuration par l'utilisateur ---
    config_name_choice = args.config_name
    if not config_name_choice:
        while config_name_choice not in ["articles", "publications"]:
            try:
                config_name_choice = input("Entrez la configuration à traiter ('articles' ou 'publications'): ").strip().lower()
            except KeyboardInterrupt:
                logger.info("\nOpération annulée par l'utilisateur.")
                return
            except EOFError:
                logger.error("\nEntrée non attendue. Arrêt.")
                return
    logger.info(f"Configuration sélectionnée: {config_name_choice}")

    # --- Authentification avec le Hub Hugging Face ---
    hf_token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not hf_token:
        logger.info("Token Hugging Face non trouvé. Tentative de connexion interactive.")
        try:
            login()
            hf_token = HfFolder.get_token()
            if not hf_token:
                 logger.error("Connexion interactive échouée ou token non sauvegardé. Veuillez vous connecter manuellement via `huggingface-cli login`.")
                 return
        except Exception as e:
            logger.error(f"Erreur lors de la connexion interactive à Hugging Face: {e}")
            return
    logger.info("Authentification Hugging Face réussie.")

    # --- Chargement du cache ---
    cache_data = load_cache(cache_file_path, logger)
    logger.info(f"{len(cache_data)} éléments chargés depuis le cache.")

    # --- Chargement du dataset ---
    logger.info(f"Chargement du dataset '{repo_id}', configuration '{config_name_choice}'...")
    try:
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=hf_token)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")
    logger.info(f"Colonnes disponibles: {ds.column_names}")

    if text_column_name not in ds.column_names:
        logger.error(f"La colonne texte '{text_column_name}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return
    
    if id_column_name not in ds.column_names:
        logger.error(f"La colonne ID '{id_column_name}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return

    # --- Application de l'analyse ---
    logger.info(f"Début de l'analyse {model_choice.upper()} pour la colonne '{text_column_name}'...")
    
    fn_kwargs_for_map = {
        "model_choice": model_choice,
        "api_key": api_key,
        "model_name": model_name,
        "cache": cache_data,
        "cache_file_path": cache_file_path,
        "logger": logger,
        "text_column_name": text_column_name,
        "id_column_name": id_column_name
    }

    ds_processed = ds.map(
        process_batch_with_analysis,
        batched=True,
        batch_size=batch_size,
        fn_kwargs=fn_kwargs_for_map,
        desc=f"Analyse {model_choice.upper()} (col: {text_column_name})"
    )
    
    logger.info(f"Analyse {model_choice.upper()} terminée.")
    save_cache(cache_file_path, cache_data, logger)
    logger.info(f"Cache final sauvegardé. Total d'éléments dans le cache: {len(cache_data)}")

    # --- Réorganisation des colonnes ---
    new_cols = [
        f"{model_choice}_centralite_islam_musulmans", f"{model_choice}_centralite_justification",
        f"{model_choice}_subjectivite_score", f"{model_choice}_subjectivite_justification",
        f"{model_choice}_polarite", f"{model_choice}_polarite_justification"
    ]
    
    insert_after_col = "sentiment_score"
    
    current_columns = list(ds_processed.column_names)
    logger.info(f"Colonnes actuelles avant réorganisation: {current_columns}")

    if insert_after_col in current_columns:
        insert_idx = current_columns.index(insert_after_col) + 1
        
        # Retirer les colonnes si elles existent déjà
        final_columns_set = [col for col in current_columns if col not in new_cols]
        
        # Insérer les nouvelles colonnes
        ordered_columns = final_columns_set[:insert_idx] + new_cols + final_columns_set[insert_idx:]
        
        if set(ordered_columns) == set(current_columns):
            ds_processed = ds_processed.select_columns(ordered_columns)
            logger.info(f"Colonnes réorganisées. Nouvel ordre: {ds_processed.column_names}")
        else:
            logger.warning(f"La réorganisation des colonnes a échoué à maintenir toutes les colonnes. Ordre inchangé.")
    else:
        logger.warning(f"La colonne '{insert_after_col}' n'a pas été trouvée. Les nouvelles colonnes seront ajoutées à la fin.")

    # Afficher un aperçu des nouvelles colonnes
    num_examples_to_show = min(5, len(ds_processed))
    if num_examples_to_show > 0:
        for col_name in new_cols:
            if col_name in ds_processed.column_names:
                 logger.info(f"Aperçu (premiers {num_examples_to_show}) pour '{col_name}': {ds_processed[col_name][:num_examples_to_show]}")

    # --- Sauvegarde du dataset traité sur le Hub ---
    logger.info(f"Sauvegarde du dataset traité vers le Hub Hugging Face (repo: '{repo_id}', config: '{config_name_choice}')...")
    try:
        ds_processed.push_to_hub(repo_id, config_name=config_name_choice, token=hf_token, max_shard_size=max_shard_size)
        logger.info("Dataset traité et sauvegardé avec succès sur le Hub.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du dataset sur le Hub: {e}")
        logger.error("Le dataset traité est disponible localement mais n'a pas été poussé.")

if __name__ == "__main__":
    main()