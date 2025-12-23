#!/usr/bin/env python3
"""
calculate_sentiment_AI.py
===========================

Ajoute des colonnes avec l'analyse de sentiment détaillée via l'API Gemini, ChatGPT ou Mistral
à un dataset Hugging Face existant. Ce script se base sur une colonne texte
spécifiée (par défaut 'OCR').

Le script :
1. Charge un dataset Hugging Face.
2. Pour chaque texte dans la colonne spécifiée :
    a. Vérifie si une analyse existe dans un cache local.
    b. Si non, appelle l'API Gemini, ChatGPT ou Mistral avec un prompt structuré.
    c. Sauvegarde le résultat de l'API dans le cache.
3. Ajoute les résultats de l'analyse dans de nouvelles colonnes
   (préfixées par "gemini_", "chatgpt_" ou "mistral_").
4. Réorganise les colonnes pour placer les nouvelles colonnes après
   la colonne 'sentiment_score' (supposée exister).
5. Pousse le dataset modifié vers le Hugging Face Hub.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications'),
le modèle d'IA à utiliser ('gemini', 'chatgpt' ou 'mistral'), et peut spécifier le nom du repo,
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
MISTRAL_API_KEY  Clé API pour Mistral AI.
HF_TOKEN         Jeton d'accès personnel pour le Hugging Face Hub.

Dépendances supplémentaires
-------------------------
    pip install datasets huggingface_hub google-genai python-dotenv openai mistralai pydantic rich
"""
import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

from datasets import load_dataset, Dataset
from huggingface_hub import get_token, login
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Rich for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm
from rich import box

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    from google.genai import errors
except ImportError:
    print("Veuillez installer le SDK Google GenAI: pip install google-genai")
    genai = None

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("Veuillez installer la librairie OpenAI: pip install openai")
    OpenAI = None

# Mistral AI SDK
try:
    from mistralai import Mistral
except ImportError:
    print("Veuillez installer la librairie Mistral AI: pip install mistralai")
    Mistral = None

# Global Rich console
console = Console()

# Configuration du logging avec Rich
def configure_logging() -> logging.Logger:
    """Configure le logging avec Rich pour un affichage élégant."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )
    return logging.getLogger(__name__)

# Pydantic model for structured outputs (used by both Gemini and ChatGPT)
class SentimentAnalysisOutput(BaseModel):
    """Schema for sentiment analysis output - used for structured outputs with AI APIs."""
    centralite_islam_musulmans: Literal[
        "Très central", "Central", "Secondaire", "Marginal", "Non abordé"
    ] = Field(description="Importance accordée aux thèmes liés à l'islam et aux musulmans dans l'article")
    centralite_justification: str = Field(description="Courte justification en 1 phrase sur la centralité de l'islam/des musulmans")
    subjectivite_score: Optional[int] = Field(
        default=None,
        description="Score de subjectivité de 1 à 5, ou null si le sujet n'est pas abordé",
        ge=1, le=5
    )
    subjectivite_justification: str = Field(description="Justification en 1-2 phrases pour le score de subjectivité")
    polarite: Literal[
        "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable"
    ] = Field(description="Sentiment général exprimé dans l'article envers l'islam et/ou les musulmans")
    polarite_justification: str = Field(description="Justification en 1-2 phrases pour la polarité")


def validate_sentiment_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide la sortie de l'analyse de sentiment en utilisant le modèle Pydantic."""
    try:
        validated = SentimentAnalysisOutput(**data)
        return validated.model_dump()
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")

# --- Model Configuration ---
# Gemini 3 Flash Preview with thinking_level support (use "low" for minimal latency)
GEMINI_MODEL_NAME = "gemini-3-flash-preview"
# GPT-5 Mini - latest efficient model with structured output support
CHATGPT_MODEL_NAME = "gpt-5-mini"
# Ministral 3 14B (2025-12) - efficient 14B parameter model with structured output support
MISTRAL_MODEL_NAME = "ministral-3-14b-latest"

# --- Cache Configuration ---
GEMINI_CACHE_FILE_DEFAULT_NAME = "gemini_sentiment_cache.json"
CHATGPT_CACHE_FILE_DEFAULT_NAME = "chatgpt_sentiment_cache.json"
MISTRAL_CACHE_FILE_DEFAULT_NAME = "mistral_sentiment_cache.json"

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

# Prompt pour l'analyse de sentiment - loaded from markdown file
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

# Load once at module level - used as system instruction for both APIs
SYSTEM_INSTRUCTION = _load_base_prompt()


def create_user_prompt_for_structured(article_text: str) -> str:
    """Create the user prompt with the article text to analyze."""
    return f"Texte à analyser:\n---\n{article_text}\n---"

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
    Utilise les structured outputs avec un schema Pydantic et system_instruction.
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

    # Use structured output with Pydantic schema and system instruction
    user_prompt = create_user_prompt_for_structured(article_text)
    
    # Configure with thinking_level="low" for minimal reasoning overhead (Gemini 3 models)
    # System instruction contains the full analysis prompt
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=SentimentAnalysisOutput,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_level="low")
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=generation_config
            )

            # With structured outputs, response.parsed contains the Pydantic model instance
            if response.parsed:
                validated_output = response.parsed.model_dump()
                return {**validated_output, "analysis_error": None}
            
            # Fallback to text parsing if parsed is not available
            if not response.text:
                logger.warning(f"Réponse vide de Gemini pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": "Réponse vide de Gemini après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                json_response = json.loads(response.text)
                validated_output = validate_sentiment_output(json_response)
                return {**validated_output, "analysis_error": None}
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON de la réponse Gemini (essai {attempt + 1}/{max_retries}): {e}. Réponse: {response.text[:500]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"JSONDecodeError: {e}", "raw_response_snippet": response.text[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
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
    Analyse le sentiment d'un texte d'article en utilisant l'API ChatGPT avec structured outputs.
    Utilise client.chat.completions.parse() avec un modèle Pydantic pour la validation automatique.
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

    # Use structured outputs with chat.completions.parse()
    user_prompt = create_user_prompt_for_structured(article_text)

    for attempt in range(max_retries):
        try:
            # Use chat.completions.parse() for structured outputs with Pydantic
            # Note: GPT-5 Mini only supports default temperature (1)
            completion = client.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SentimentAnalysisOutput
            )

            message = completion.choices[0].message
            
            # Check for refusal
            if message.refusal:
                logger.warning(f"ChatGPT a refusé la requête (essai {attempt + 1}/{max_retries}): {message.refusal}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"Refusal: {message.refusal}"}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            # With structured outputs, message.parsed contains the Pydantic model instance
            if message.parsed:
                validated_output = message.parsed.model_dump()
                return {**validated_output, "analysis_error": None}
            
            # Fallback to content parsing if parsed is not available
            if not message.content:
                logger.warning(f"Réponse vide de ChatGPT pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": "Réponse vide de ChatGPT après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                json_payload = json.loads(message.content)
                validated_output = validate_sentiment_output(json_payload)
                return {**validated_output, "analysis_error": None}
            except json.JSONDecodeError as je:
                logger.error(f"Échec parsing JSON (essai {attempt + 1}/{max_retries}): {je}. Extrait: {message.content[:200]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"JSON parse error: {je}", "raw_response_snippet": message.content[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            except (ValueError, KeyError) as ve:
                logger.error(f"Validation échouée (essai {attempt + 1}/{max_retries}): {ve}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"ValidationError: {ve}", "parsed_data": json_payload}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

        except Exception as e:
            logger.error(f"Erreur lors de l'appel à ChatGPT (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "analysis_error": f"Exception: {e}"}
            time.sleep(initial_backoff * (2 ** attempt))

    return {**default_error_result, "analysis_error": "Échec de l'analyse ChatGPT après plusieurs tentatives."}


def analyze_text_with_mistral(
    article_text: str,
    mistral_api_key: str,
    model_name: str,
    logger: logging.Logger,
    max_retries: int = 3,
    initial_backoff: int = 5
) -> Dict[str, Any]:
    """
    Analyse le sentiment d'un texte d'article en utilisant l'API Mistral avec structured outputs.
    Utilise client.chat.parse() avec un modèle Pydantic pour la validation automatique.
    Retourne un dictionnaire avec les champs de SentimentAnalysisOutput et un champ 'analysis_error'.
    """
    default_error_result = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": "Erreur lors de l'analyse Mistral.",
        "subjectivite_score": None,
        "subjectivite_justification": "Erreur lors de l'analyse Mistral.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": "Erreur lors de l'analyse Mistral.",
        "analysis_error": "Erreur inconnue"
    }

    if not article_text or not article_text.strip():
        logger.warning("Texte de l'article vide ou manquant pour l'analyse Mistral.")
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
        client = Mistral(api_key=mistral_api_key)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Mistral: {e}")
        return {**default_error_result, "analysis_error": f"Erreur client Mistral: {e}"}

    # Use structured outputs with chat.complete() - system instruction in messages
    user_prompt = create_user_prompt_for_structured(article_text)

    for attempt in range(max_retries):
        try:
            # Use chat.complete() with response_format as dict containing the Pydantic model
            # The Mistral SDK will automatically convert the Pydantic model to JSON schema
            completion = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sentiment_analysis",
                        "schema": SentimentAnalysisOutput.model_json_schema(),
                        "strict": True
                    }
                },
                max_tokens=512,
                temperature=0.2
            )

            message = completion.choices[0].message
            
            # With structured outputs via chat.complete(), the response is in message.content as JSON
            if not message.content:
                logger.warning(f"Réponse vide de Mistral pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": "Réponse vide de Mistral après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                json_payload = json.loads(message.content)
                validated_output = validate_sentiment_output(json_payload)
                return {**validated_output, "analysis_error": None}
            except json.JSONDecodeError as je:
                logger.error(f"Échec parsing JSON Mistral (essai {attempt + 1}/{max_retries}): {je}. Extrait: {message.content[:200]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"JSON parse error: {je}", "raw_response_snippet": message.content[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            except (ValueError, KeyError) as ve:
                logger.error(f"Validation Mistral échouée (essai {attempt + 1}/{max_retries}): {ve}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"ValidationError: {ve}", "parsed_data": json_payload}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erreur lors de l'appel à Mistral (essai {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check for specific Mistral API errors
            if "Unexpected type" in error_msg:
                logger.error("Erreur de type: vérifiez le format de response_format")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"Type error in API call: {error_msg}"}
            elif hasattr(e, 'status_code'):
                logger.error(f"HTTP status: {e.status_code}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "analysis_error": f"HTTP {e.status_code}: {error_msg}"}
            
            if attempt == max_retries - 1:
                return {**default_error_result, "analysis_error": f"Exception: {error_msg}"}
            time.sleep(initial_backoff * (2 ** attempt))

    return {**default_error_result, "analysis_error": "Échec de l'analyse Mistral après plusieurs tentatives."}


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
                elif model_choice == "chatgpt":
                    analysis_result = analyze_text_with_chatgpt(text, api_key, model_name, logger)
                else:  # mistral
                    analysis_result = analyze_text_with_mistral(text, api_key, model_name, logger)
                cache[cache_key] = analysis_result
                processed_in_batch += 1
            else:
                analysis_result = cached_result
                logger.debug(f"Résultat trouvé dans le cache pour l'ID '{cache_key}'.")
        else:
            logger.debug(f"Analyse {model_choice.upper()} pour l'ID '{cache_key}' (texte début): {text[:50]}...")
            if model_choice == "gemini":
                analysis_result = analyze_text_with_gemini(text, api_key, model_name, logger)
            elif model_choice == "chatgpt":
                analysis_result = analyze_text_with_chatgpt(text, api_key, model_name, logger)
            else:  # mistral
                analysis_result = analyze_text_with_mistral(text, api_key, model_name, logger)
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
    
    # Afficher le titre
    console.print(Panel.fit(
        "[bold cyan]Analyse de Sentiment AI[/bold cyan]\n"
        "[dim]Gemini / ChatGPT pour les représentations de l'islam[/dim]",
        border_style="cyan"
    ))
    
    # Déterminer le répertoire du script et la racine du projet
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Charger les variables d'environnement depuis .env à la racine du projet
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        console.print(f"[green]✓[/green] Variables d'environnement chargées depuis {dotenv_path}")
    else:
        console.print(f"[yellow]⚠[/yellow] Fichier .env non trouvé à {dotenv_path}")

    parser = argparse.ArgumentParser(description="Ajoute des colonnes d'analyse de sentiment via Gemini, ChatGPT ou Mistral à un dataset Hugging Face.")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset).")
    parser.add_argument("--config-name", type=str, default=None, help="Nom de la configuration à traiter (ex: 'articles', 'publications'). Sera demandé si non fourni.")
    parser.add_argument("--text-column", default="OCR", help="Nom de la colonne contenant le texte à analyser.")
    parser.add_argument("--id-column", default="o:id", help="Nom de la colonne contenant les identifiants uniques pour le cache.")
    parser.add_argument("--model", type=str, default=None, help="Modèle à utiliser ('gemini', 'chatgpt' ou 'mistral'). Sera demandé si non fourni.")
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
        model_choice = Prompt.ask(
            "[cyan]Modèle à utiliser[/cyan]",
            choices=["gemini", "chatgpt", "mistral"],
            default="gemini"
        )
    console.print(f"[blue]→[/blue] Modèle sélectionné: [bold]{model_choice.upper()}[/bold]")

    # --- Configuration selon le modèle choisi ---
    if model_choice == "gemini":
        if genai is None:
            console.print("[red]✗[/red] Le SDK Google GenAI n'est pas installé.")
            console.print("  [dim]pip install google-genai[/dim]")
            return
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]✗[/red] Variable d'environnement [bold]GOOGLE_API_KEY[/bold] non définie.")
            return
        
        model_name = GEMINI_MODEL_NAME
        cache_file_path = Path(script_dir / GEMINI_CACHE_FILE_DEFAULT_NAME)
        
        # Test du client
        with console.status("[bold green]Vérification de la clé API Google...", spinner="dots"):
            try:
                _ = genai.Client(api_key=api_key)
            except Exception as e:
                console.print(f"[red]✗[/red] Erreur client Gemini: {e}")
                return
        console.print(f"[green]✓[/green] Clé API Google valide. Modèle: [bold]{model_name}[/bold]")
            
    elif model_choice == "chatgpt":
        if OpenAI is None:
            console.print("[red]✗[/red] Le SDK OpenAI n'est pas installé.")
            console.print("  [dim]pip install openai[/dim]")
            return
        
        api_key = os.getenv("CHATGPT")
        if not api_key:
            console.print("[red]✗[/red] Variable d'environnement [bold]CHATGPT[/bold] non définie.")
            return
        
        model_name = CHATGPT_MODEL_NAME
        cache_file_path = Path(script_dir / CHATGPT_CACHE_FILE_DEFAULT_NAME)
        
        # Test du client
        with console.status("[bold green]Vérification de la clé API OpenAI...", spinner="dots"):
            try:
                _ = OpenAI(api_key=api_key)
            except Exception as e:
                console.print(f"[red]✗[/red] Erreur client ChatGPT: {e}")
                return
        console.print(f"[green]✓[/green] Clé API OpenAI valide. Modèle: [bold]{model_name}[/bold]")

    else:  # mistral
        if Mistral is None:
            console.print("[red]✗[/red] Le SDK Mistral AI n'est pas installé.")
            console.print("  [dim]pip install mistralai[/dim]")
            return
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            console.print("[red]✗[/red] Variable d'environnement [bold]MISTRAL_API_KEY[/bold] non définie.")
            return
        
        model_name = MISTRAL_MODEL_NAME
        cache_file_path = Path(script_dir / MISTRAL_CACHE_FILE_DEFAULT_NAME)
        
        # Test du client
        with console.status("[bold green]Vérification de la clé API Mistral...", spinner="dots"):
            try:
                _ = Mistral(api_key=api_key)
            except Exception as e:
                console.print(f"[red]✗[/red] Erreur client Mistral: {e}")
                return
        console.print(f"[green]✓[/green] Clé API Mistral valide. Modèle: [bold]{model_name}[/bold]")

    # --- Choix de la configuration par l'utilisateur ---
    config_name_choice = args.config_name
    if not config_name_choice:
        config_name_choice = Prompt.ask(
            "[cyan]Configuration à traiter[/cyan]",
            choices=["articles", "publications"],
            default="articles"
        )
    console.print(f"[blue]→[/blue] Configuration: [bold]{config_name_choice}[/bold]")

    # --- Authentification avec le Hub Hugging Face ---
    hf_token = os.getenv("HF_TOKEN") or get_token()
    if not hf_token:
        console.print("[yellow]ℹ[/yellow] Token Hugging Face non trouvé. Connexion interactive...")
        try:
            login()
            hf_token = get_token()
            if not hf_token:
                console.print("[red]✗[/red] Connexion échouée. Utilisez: [dim]huggingface-cli login[/dim]")
                return
        except Exception as e:
            console.print(f"[red]✗[/red] Erreur de connexion HF: {e}")
            return
    console.print("[green]✓[/green] Authentification Hugging Face réussie")

    # --- Chargement du cache ---
    cache_data = load_cache(cache_file_path, logger)
    console.print(f"[green]✓[/green] Cache chargé: [bold]{len(cache_data)}[/bold] éléments")

    # --- Chargement du dataset ---
    with console.status(f"[bold green]Chargement du dataset '{config_name_choice}'...", spinner="dots"):
        try:
            ds = load_dataset(repo_id, name=config_name_choice, split="train", token=hf_token)
        except Exception as e:
            console.print(f"[red]✗[/red] Erreur de chargement: {e}")
            return

    # Afficher les infos du dataset dans un tableau
    info_table = Table(title="Dataset Info", box=box.ROUNDED)
    info_table.add_column("Propriété", style="cyan")
    info_table.add_column("Valeur", style="green")
    info_table.add_row("Repository", repo_id)
    info_table.add_row("Configuration", config_name_choice)
    info_table.add_row("Nombre de lignes", str(len(ds)))
    info_table.add_row("Colonne texte", text_column_name)
    info_table.add_row("Colonne ID", id_column_name)
    console.print(info_table)

    if text_column_name not in ds.column_names:
        console.print(f"[red]✗[/red] Colonne texte '{text_column_name}' introuvable")
        console.print(f"  [dim]Disponibles: {ds.column_names}[/dim]")
        return
    
    if id_column_name not in ds.column_names:
        console.print(f"[red]✗[/red] Colonne ID '{id_column_name}' introuvable")
        console.print(f"  [dim]Disponibles: {ds.column_names}[/dim]")
        return

    # --- Application de l'analyse ---
    console.print(f"\n[bold cyan]Début de l'analyse {model_choice.upper()}[/bold cyan]")
    
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
        desc=f"Analyse {model_choice.upper()}",
        num_proc=1  # Disable multiprocessing to avoid Pydantic pickling warnings
    )
    
    console.print(f"[green]✓[/green] Analyse {model_choice.upper()} terminée")
    save_cache(cache_file_path, cache_data, logger)
    console.print(f"[green]✓[/green] Cache sauvegardé: [bold]{len(cache_data)}[/bold] éléments")

    # --- Réorganisation des colonnes ---
    new_cols = [
        f"{model_choice}_centralite_islam_musulmans", f"{model_choice}_centralite_justification",
        f"{model_choice}_subjectivite_score", f"{model_choice}_subjectivite_justification",
        f"{model_choice}_polarite", f"{model_choice}_polarite_justification"
    ]
    
    insert_after_col = "sentiment_score"
    
    current_columns = list(ds_processed.column_names)

    if insert_after_col in current_columns:
        insert_idx = current_columns.index(insert_after_col) + 1
        
        # Retirer les colonnes si elles existent déjà
        final_columns_set = [col for col in current_columns if col not in new_cols]
        
        # Insérer les nouvelles colonnes
        ordered_columns = final_columns_set[:insert_idx] + new_cols + final_columns_set[insert_idx:]
        
        if set(ordered_columns) == set(current_columns):
            ds_processed = ds_processed.select_columns(ordered_columns)
            console.print("[green]✓[/green] Colonnes réorganisées")
        else:
            console.print("[yellow]⚠[/yellow] Réorganisation des colonnes échouée, ordre inchangé")
    else:
        console.print(f"[yellow]⚠[/yellow] Colonne '{insert_after_col}' non trouvée, nouvelles colonnes ajoutées à la fin")

    # Afficher un aperçu des résultats dans un tableau
    num_examples = min(3, len(ds_processed))
    if num_examples > 0:
        preview_table = Table(title=f"Aperçu des résultats ({num_examples} premiers)", box=box.ROUNDED)
        preview_table.add_column("ID", style="dim")
        preview_table.add_column("Centralité", style="cyan")
        preview_table.add_column("Subj.", style="yellow", justify="center")
        preview_table.add_column("Polarité", style="green")
        
        for i in range(num_examples):
            preview_table.add_row(
                str(ds_processed[id_column_name][i]),
                str(ds_processed[f"{model_choice}_centralite_islam_musulmans"][i] or "-")[:20],
                str(ds_processed[f"{model_choice}_subjectivite_score"][i] or "-"),
                str(ds_processed[f"{model_choice}_polarite"][i] or "-")[:15]
            )
        console.print(preview_table)

    # --- Sauvegarde du dataset traité sur le Hub ---
    console.print(f"\n[bold cyan]Sauvegarde vers Hugging Face Hub[/bold cyan]")
    with console.status("[bold green]Push en cours...", spinner="dots"):
        try:
            ds_processed.push_to_hub(repo_id, config_name=config_name_choice, token=hf_token, max_shard_size=max_shard_size)
        except Exception as e:
            console.print(f"[red]✗[/red] Erreur lors du push: {e}")
            console.print("[yellow]ℹ[/yellow] Le dataset traité est disponible localement.")
            return
    
    # Résumé final
    console.print(Panel.fit(
        f"[bold green]✓ Traitement terminé avec succès![/bold green]\n\n"
        f"[cyan]Repository:[/cyan] {repo_id}\n"
        f"[cyan]Configuration:[/cyan] {config_name_choice}\n"
        f"[cyan]Lignes traitées:[/cyan] {len(ds_processed)}\n"
        f"[cyan]Modèle:[/cyan] {model_name}",
        title="Résumé",
        border_style="green"
    ))

if __name__ == "__main__":
    main()