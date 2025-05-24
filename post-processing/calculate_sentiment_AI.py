#!/usr/bin/env python3
"""
calculate_sentiment_AI.py
===========================

Ajoute des colonnes avec l'analyse de sentiment détaillée via l'API Gemini
à un dataset Hugging Face existant. Ce script se base sur une colonne texte
spécifiée (par défaut 'OCR').

Le script :
1. Charge un dataset Hugging Face.
2. Pour chaque texte dans la colonne spécifiée :
    a. Vérifie si une analyse existe dans un cache local.
    b. Si non, appelle l'API Gemini avec un prompt structuré.
    c. Sauvegarde le résultat de l'API dans le cache.
3. Ajoute les résultats de l'analyse Gemini dans de nouvelles colonnes
   (préfixées par "gemini_").
4. Réorganise les colonnes pour placer les nouvelles colonnes Gemini après
   la colonne 'sentiment_score' (supposée exister).
5. Pousse le dataset modifié vers le Hugging Face Hub.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications')
et peut spécifier le nom du repo, la colonne texte, etc., via des arguments CLI.

Usage
-----
    python post-processing/calculate_sentiment_AI.py [--repo MON_USER/MON_DATASET] [--config-name CONFIG] [--text-column TEXT_COL]

Exemple:
    python post-processing/calculate_sentiment_AI.py --repo fmadore/iwac-newspaper-articles --config-name articles

Variables d'environnement
-------------------------
GOOGLE_API_KEY   Clé API pour Google Gemini.
HF_TOKEN         Jeton d'accès personnel pour le Hugging Face Hub.

Dépendances supplémentaires
-------------------------
    pip install datasets huggingface_hub google-api-python-client pydantic python-dotenv tqdm google-ai-generativelanguage
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
from pydantic import BaseModel, Field, ValidationError
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
    exit(1)

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

# Modèle Pydantic pour la sortie structurée de Gemini
class SentimentAnalysisOutput(BaseModel):
    centralite_islam_musulmans: str
    centralite_justification: str
    subjectivite_score: Optional[int] = Field(default=None)
    subjectivite_justification: str
    polarite: str
    polarite_justification: str

# --- Gemini Configuration ---
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Model name to be used

# --- Cache Configuration ---
CACHE_FILE_DEFAULT_NAME = "gemini_sentiment_cache.json"

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

# Prompt pour l'analyse de sentiment Gemini (repris de votre script)
def create_gemini_prompt(article_text: str) -> str:
    prompt = f'''
    Vous êtes un expert en analyse de sentiments, spécialisé dans l'étude des représentations de l'islam et des musulmans dans les médias, notamment en Afrique de l'Ouest francophone. Votre tâche est d'analyser le texte fourni sous cet angle spécifique et de renvoyer une analyse structurée en JSON.

    Votre analyse doit spécifiquement évaluer comment l'islam et/ou les musulmans sont dépeints ou représentés dans l'article. La subjectivité et la polarité doivent être jugées par rapport à cette représentation. Si l'islam et les musulmans ne sont qu'un sujet marginal ou non pertinent dans l'article, indiquez-le clairement.

    Pour le texte de l'article suivant :
    ---
    {article_text}
    ---

    Veuillez fournir les informations suivantes au format JSON respectant le schéma Pydantic SentimentAnalysisOutput:
    {{
      "centralite_islam_musulmans": "<Très central | Central | Secondaire | Marginal | Non abordé>",
      "centralite_justification": "<Courte justification (1 phrase) expliquant le niveau de centralité de l'islam/des musulmans dans l'article>",
      "subjectivite_score": <score_de_1_a_5_ou_null_si_non_aborde>,
      "subjectivite_justification": "<justification_en_1_2_phrases expliquant pourquoi ce score de subjectivité a été attribué concernant la manière dont l'article traite de l'islam et/ou des musulmans, ou 'Non applicable si le sujet n'est pas abordé'>",
      "polarite": "<Très positif | Positif | Neutre | Négatif | Très négatif | Non applicable>",
      "polarite_justification": "<justification_en_1_2_phrases expliquant pourquoi cette polarité a été attribuée en ce qui concerne le portrait de l'islam et/ou des musulmans dans l'article, ou 'Non applicable si le sujet n'est pas abordé'>"
    }}

    Voici les barèmes à utiliser :

    Centralité de l'islam et des musulmans dans l'article :
    - Très central : L'article est principalement ou entièrement consacré à l'islam et/ou aux musulmans.
    - Central : L'islam et/ou les musulmans sont un des sujets principaux de l'article.
    - Secondaire : L'islam et/ou les musulmans sont mentionnés ou discutés, mais ne constituent pas le focus principal de l'article.
    - Marginal : L'islam et/ou les musulmans sont brièvement mentionnés de manière anecdotique ou périphérique.
    - Non abordé : L'article ne traite pas du tout de l'islam ou des musulmans.

    Subjectivité (note de 1 à 5) – Évaluez le degré d'objectivité/subjectivité de l'article DANS SA MANIÈRE DE REPRÉSENTER l'islam et/ou les musulmans (Attribuez 'null' si 'Non abordé' pour la centralité) :
    1 : Très objectif (rapporte des faits vérifiables sur l'islam/les musulmans sans exprimer d'opinions ou de sentiments personnels à leur sujet, style purement informatif sur ce thème).
    2 : Plutôt objectif (principalement factuel concernant l'islam/les musulmans, mais peut contenir des traces subtiles d'opinions ou des choix de mots suggérant une perspective limitée sur ce thème).
    3 : Mixte (contient un mélange équilibré de faits et d'opinions/sentiments personnels concernant l'islam/les musulmans, ou présente plusieurs points de vue sur ce thème).
    4 : Plutôt subjectif (exprime clairement des opinions, des sentiments ou des jugements sur l'islam/les musulmans, même s'il s'appuie sur certains faits pour les étayer).
    5 : Très subjectif (fortement biaisé dans sa représentation de l'islam/des musulmans, exprime des opinions et des émotions intenses à leur sujet, avec peu ou pas de présentation objective des faits, style éditorial ou billet d'humeur sur ce thème).

    Polarité – Évaluez le sentiment général exprimé DANS L'ARTICLE ENVERS l'islam et/ou les musulmans, ou concernant leur représentation (Attribuez 'Non applicable' si 'Non abordé' pour la centralité) :
    - Très positif : Le portrait de l'islam/des musulmans est extrêmement favorable, enthousiaste, élogieux.
    - Positif : Le portrait de l'islam/des musulmans est favorable, optimiste.
    - Neutre : Pas de sentiment clair envers l'islam/des musulmans ou équilibre entre aspects positifs et négatifs dans leur représentation ; ton factuel sans charge émotionnelle marquée à leur égard.
    - Négatif : Le portrait de l'islam/des musulmans est défavorable, critique, pessimiste.
    - Très négatif : Le portrait de l'islam/des musulmans est extrêmement défavorable, alarmiste, très critique.

    Si la centralité est "Non abordé", le "subjectivite_score" doit être null, et "polarite", "subjectivite_justification", et "polarite_justification" doivent être "Non applicable". Le JSON doit toujours être valide.
    Par exemple, si "centralite_islam_musulmans" est "Non abordé":
    {{
      "centralite_islam_musulmans": "Non abordé",
      "centralite_justification": "L'article ne mentionne ni l'islam ni les musulmans.",
      "subjectivite_score": null,
      "subjectivite_justification": "Non applicable car le sujet n'est pas abordé.",
      "polarite": "Non applicable",
      "polarite_justification": "Non applicable car le sujet n'est pas abordé."
    }}

    Assurez-vous que votre réponse est uniquement le JSON structuré demandé, sans texte ou formatage supplémentaire avant ou après le JSON.
    '''
    return prompt

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
    Retourne un dictionnaire avec les champs de SentimentAnalysisOutput et un champ 'gemini_analysis_error'.
    """
    default_error_result = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": "Erreur lors de l'analyse Gemini.",
        "subjectivite_score": None,
        "subjectivite_justification": "Erreur lors de l'analyse Gemini.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": "Erreur lors de l'analyse Gemini.",
        "gemini_analysis_error": "Erreur inconnue"
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
            "gemini_analysis_error": "Texte vide fourni pour analyse"
        }

    try:
        client = genai.Client(api_key=google_api_key)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Gemini: {e}")
        return {**default_error_result, "gemini_analysis_error": f"Erreur client Gemini: {e}"}

    prompt_content = create_gemini_prompt(article_text)
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_content)]
        )
    ]

    # Configuration pour la génération de contenu, demandant un JSON structuré
    # selon le modèle Pydantic.
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2, # Température basse pour une sortie plus déterministe
        response_schema=SentimentAnalysisOutput 
    )
    
    # Safety settings (optional, adjust as needed, example from google.generativeai, might differ for google.genai)
    # For google.genai, safety settings are often part of the model resource or request.
    # If direct equivalent is not obvious, start without or consult google.genai specific docs.
    # For now, omitting direct safety_settings in the call, assuming defaults or client-level config.

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=f"models/{model_name}", # Ensure model name is prefixed with "models/"
                contents=contents,
                generation_config=generation_config
            )

            if not response.text:
                logger.warning(f"Réponse vide de Gemini pour le texte (essai {attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    return {**default_error_result, "gemini_analysis_error": "Réponse vide de Gemini après plusieurs essais."}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                # response.text should be a JSON string if response_mime_type="application/json"
                # and response_schema is used.
                json_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON de la réponse Gemini (essai {attempt + 1}/{max_retries}): {e}. Réponse: {response.text[:500]}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "gemini_analysis_error": f"JSONDecodeError: {e}", "raw_response_snippet": response.text[:500]}
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            
            try:
                # Pydantic validation (even if response_schema is used, good for robustness)
                validated_output = SentimentAnalysisOutput(**json_response)
                return {**validated_output.model_dump(), "gemini_analysis_error": None}
            except ValidationError as e:
                logger.error(f"Erreur de validation Pydantic (essai {attempt + 1}/{max_retries}): {e}. Données reçues: {json_response}")
                if attempt == max_retries - 1:
                    return {**default_error_result, "gemini_analysis_error": f"Pydantic ValidationError: {e}", "parsed_data": json_response}
                time.sleep(initial_backoff * (2 ** attempt))
                continue

        except errors.DeadlineExceededError as e: # Specific to google.genai.errors
            logger.warning(f"Timeout Gemini (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "gemini_analysis_error": f"DeadlineExceededError: {e}"}
            time.sleep(initial_backoff * (2 ** attempt) * 2)
        except errors.ResourceExhaustedError as e: # Specific to google.genai.errors (rate limiting)
            logger.warning(f"Quota Gemini/limite de taux (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "gemini_analysis_error": f"ResourceExhaustedError: {e}"}
            time.sleep(initial_backoff * (2 ** attempt) * 5)
        # google.genai.errors might not have BlockedPromptException directly.
        # It might be a generic APIError with details.
        except errors.APIError as e: # Catch other specific API errors from google.genai
            # Check for prompt feedback or specific error details if available
            # Example: if e.details and 'PROMPT_BLOCKED' in str(e.details):
            logger.error(f"Erreur API Gemini (essai {attempt + 1}/{max_retries}): {e}. Texte (début): {article_text[:200]}")
            # You might want to inspect e.error_code or e.message for more details
            error_message = f"APIError: {e}"
            # Attempt to extract prompt feedback if available (structure might vary)
            # if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
            #    error_message += f" Prompt Feedback: {e.response.prompt_feedback}"
            if attempt == max_retries - 1:
                return {**default_error_result, "gemini_analysis_error": error_message}
            time.sleep(initial_backoff * (2 ** attempt))
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Erreur inattendue lors de l'appel à Gemini (essai {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {**default_error_result, "gemini_analysis_error": f"Exception: {e}"}
            time.sleep(initial_backoff * (2 ** attempt))
            
    return {**default_error_result, "gemini_analysis_error": "Échec de l'analyse Gemini après plusieurs tentatives."}


def process_batch_with_gemini_analysis(
    batch: Dict[str, List[Any]], 
    google_api_key: str, # Pass API key
    model_name: str,     # Pass model name
    cache: Dict[str, Any], 
    cache_file_path: Path, 
    logger: logging.Logger,
    text_column_name: str
) -> Dict[str, List[Any]]:
    """
    Applique l'analyse de sentiment Gemini à un batch d'exemples, utilisant et mettant à jour un cache.
    """
    ocr_texts = batch[text_column_name]
    
    # Initialiser les listes pour les nouvelles colonnes
    results_centralite_islam_musulmans = []
    results_centralite_justification = []
    results_subjectivite_score = []
    results_subjectivite_justification = []
    results_polarite = []
    results_polarite_justification = []
    # results_gemini_analysis_error = [] # Supprimé

    processed_in_batch = 0
    for text in ocr_texts:
        # Utiliser le texte OCR comme clé de cache. Pour des textes très longs ou non uniques,
        # une autre stratégie de clé (ex: hash, ID de ligne si disponible et unique) pourrait être envisagée.
        cache_key = str(text) # Assurer que la clé est une chaîne

        if cache_key in cache:
            analysis_result = cache[cache_key]
            logger.debug(f"Résultat trouvé dans le cache pour le texte (début): {text[:50]}...")
        else:
            logger.debug(f"Analyse Gemini pour le texte (début): {text[:50]}...")
            analysis_result = analyze_text_with_gemini(text, google_api_key, model_name, logger) # Pass key and model
            cache[cache_key] = analysis_result # analysis_result contient 'gemini_analysis_error'
            processed_in_batch +=1
        
        results_centralite_islam_musulmans.append(analysis_result.get("centralite_islam_musulmans"))
        results_centralite_justification.append(analysis_result.get("centralite_justification"))
        results_subjectivite_score.append(analysis_result.get("subjectivite_score"))
        results_subjectivite_justification.append(analysis_result.get("subjectivite_justification"))
        results_polarite.append(analysis_result.get("polarite"))
        results_polarite_justification.append(analysis_result.get("polarite_justification"))
        # results_gemini_analysis_error.append(analysis_result.get("gemini_analysis_error")) # Supprimé

    if processed_in_batch > 0 : # Save cache if new items were processed
        save_cache(cache_file_path, cache, logger)
        logger.info(f"{processed_in_batch} nouveaux éléments traités et ajoutés au cache dans ce batch.")

    batch["gemini_centralite_islam_musulmans"] = results_centralite_islam_musulmans
    batch["gemini_centralite_justification"] = results_centralite_justification
    batch["gemini_subjectivite_score"] = results_subjectivite_score
    batch["gemini_subjectivite_justification"] = results_subjectivite_justification
    batch["gemini_polarite"] = results_polarite
    batch["gemini_polarite_justification"] = results_polarite_justification
    # batch["gemini_analysis_error"] = results_gemini_analysis_error # Supprimé
    
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
        logger.warning(f"Fichier .env non trouvé à {dotenv_path}. Assurez-vous que GOOGLE_API_KEY et HF_TOKEN sont définis.")

    parser = argparse.ArgumentParser(description="Ajoute des colonnes d'analyse de sentiment Gemini à un dataset Hugging Face.")
    parser.add_argument("--repo", default="fmadore/iwac-newspaper-articles", help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset).")
    parser.add_argument("--config-name", type=str, default=None, help="Nom de la configuration à traiter (ex: 'articles', 'publications'). Sera demandé si non fourni.")
    parser.add_argument("--text-column", default="OCR", help="Nom de la colonne contenant le texte à analyser.")
    parser.add_argument("--cache-file", default=str(script_dir / CACHE_FILE_DEFAULT_NAME), help=f"Chemin vers le fichier cache JSON. Défaut: {CACHE_FILE_DEFAULT_NAME} dans le dossier du script.")
    parser.add_argument("--batch-size", type=int, default=10, help="Taille des batchs pour le traitement .map(). Attention: un batch size élevé avec des appels API peut être lent ou atteindre des limites.")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille maximale des shards Parquet lors du push vers le Hub.")
    
    args = parser.parse_args()

    repo_id = args.repo
    text_column_name = args.text_column
    cache_file_path = Path(args.cache_file)
    batch_size = args.batch_size
    max_shard_size = args.max_shard_size

    # --- Initialisation de l'API Gemini ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("La variable d'environnement GOOGLE_API_KEY n'est pas définie. Veuillez la configurer.")
        return
    
    # With google.genai, genai.configure() is not used. API key is passed to client.
    # Test basic client initialization to catch early configuration issues if desired,
    # but actual client is created in analyze_text_with_gemini.
    try:
        # Quick test if client can be nominally created (optional)
        _ = genai.Client(api_key=google_api_key) 
        logger.info(f"Clé API Google chargée. Prêt à utiliser le modèle '{GEMINI_MODEL_NAME}'.")
    except Exception as e:
        logger.error(f"Erreur lors de la pré-vérification du client Gemini avec la clé API: {e}")
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
            hf_token = HfFolder.get_token() # Re-check after login
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
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=hf_token, trust_remote_code=True)
        # Si le dataset est très gros et ne tient pas en RAM, envisager streaming=True
        # et une approche de traitement différente (pas .map directement sur tout le dataset).
        # Pour l'instant, on suppose qu'il tient en mémoire.
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")
    logger.info(f"Colonnes disponibles: {ds.column_names}")

    if text_column_name not in ds.column_names:
        logger.error(f"La colonne texte '{text_column_name}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return

    # --- Application de l'analyse Gemini ---
    logger.info(f"Début de l'analyse Gemini pour la colonne '{text_column_name}'...")
    
    # Utilisation de tqdm pour la barre de progression avec .map()
    # Note: la progression de tqdm avec .map() peut ne pas être parfaitement granulaire par item,
    # mais plutôt par batch ou chunk interne à datasets.
    
    fn_kwargs_for_map = {
        "google_api_key": google_api_key, # Pass API key
        "model_name": GEMINI_MODEL_NAME,   # Pass model name
        "cache": cache_data,
        "cache_file_path": cache_file_path,
        "logger": logger,
        "text_column_name": text_column_name
    }

    ds_processed = ds.map(
        process_batch_with_gemini_analysis,
        batched=True,
        batch_size=batch_size,
        fn_kwargs=fn_kwargs_for_map,
        desc=f"Analyse Gemini (col: {text_column_name})"
    )
    
    logger.info("Analyse Gemini terminée.")
    save_cache(cache_file_path, cache_data, logger) # Sauvegarde finale du cache
    logger.info(f"Cache final sauvegardé. Total d'éléments dans le cache: {len(cache_data)}")

    # --- Réorganisation des colonnes ---
    # Les nouvelles colonnes sont:
    # gemini_centralite_islam_musulmans, gemini_centralite_justification, 
    # gemini_subjectivite_score, gemini_subjectivite_justification,
    # gemini_polarite, gemini_polarite_justification
    
    new_gemini_cols = [
        "gemini_centralite_islam_musulmans", "gemini_centralite_justification",
        "gemini_subjectivite_score", "gemini_subjectivite_justification",
        "gemini_polarite", "gemini_polarite_justification" # "gemini_analysis_error" supprimé
    ]
    
    insert_after_col = "sentiment_score" # Colonne CamemBERT après laquelle insérer
    
    current_columns = list(ds_processed.column_names)
    logger.info(f"Colonnes actuelles avant réorganisation: {current_columns}")

    if insert_after_col in current_columns:
        insert_idx = current_columns.index(insert_after_col) + 1
        
        # Retirer les colonnes Gemini si elles existent déjà (au cas où le script est relancé sur un dataset déjà traité partiellement)
        final_columns_set = [col for col in current_columns if col not in new_gemini_cols]
        
        # Insérer les nouvelles colonnes Gemini
        ordered_columns = final_columns_set[:insert_idx] + new_gemini_cols + final_columns_set[insert_idx:]
        
        # Vérifier que toutes les colonnes originales (sauf celles remplacées) et les nouvelles sont présentes
        if set(ordered_columns) == set(current_columns): # current_columns already includes new_gemini_cols from .map
            ds_processed = ds_processed.select_columns(ordered_columns)
            logger.info(f"Colonnes réorganisées. Nouvel ordre: {ds_processed.column_names}")
        else:
            logger.warning(f"La réorganisation des colonnes a échoué à maintenir toutes les colonnes. Différence: {set(current_columns).symmetric_difference(set(ordered_columns))}. Ordre inchangé.")
            logger.warning(f"Colonnes attendues après réorganisation (théorique): {ordered_columns}")


    else:
        logger.warning(f"La colonne '{insert_after_col}' n'a pas été trouvée. Les nouvelles colonnes Gemini seront ajoutées à la fin.")
        # Pas besoin de faire select_columns si elles sont déjà à la fin (comportement par défaut de .map)

    # Afficher un aperçu des nouvelles colonnes
    num_examples_to_show = min(5, len(ds_processed))
    if num_examples_to_show > 0:
        for col_name in new_gemini_cols:
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