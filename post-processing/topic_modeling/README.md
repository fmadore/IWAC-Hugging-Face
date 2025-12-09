# Topic Modeling pour IWAC

Pipeline de modélisation de sujets avec BERTopic pour le dataset Islam West Africa Collection.

## Utilisation Simple

### Entraîner un nouveau modèle

Juste lancer le script - il va poser quelques questions simples :

```bash
python topic_modeling.py
```

Le script va :
1. Vous demander quelle configuration utiliser (articles, publications, etc.)
2. Vous demander si vous voulez entraîner un nouveau modèle ou utiliser un existant
3. Faire tout le reste automatiquement

### Ce qui est activé par défaut

✅ **Métriques de cohérence** (C_v, NPMI, U_Mass) - pour évaluer la qualité des topics  
✅ **Analyse temporelle** (topics over time) - si la colonne `pub_date` existe  
✅ **Extraction de `pub_year`** - pour faciliter l'analyse dans votre dashboard  
✅ **Sauvegarde des paramètres** - dans `training_parameters.json` pour reproductibilité  

### Options CPU (sans GPU)

Pour machines sans GPU, ajoutez `--cpu-only` :

```bash
python topic_modeling.py --cpu-only
```

### Tester avec moins de documents

Pour tester rapidement avec un sous-ensemble :

```bash
python topic_modeling.py --cpu-only --max-documents 1000
```

## Résultats

### Colonnes ajoutées au dataset HuggingFace

- `topic_id` : ID du topic assigné (-1 = outlier)
- `topic_prob` : Probabilité du topic assigné
- `topic_label` : Label lisible du topic

### Fichiers sauvegardés localement

Dans le dossier `bertopic_model/` :
- `training_parameters.json` : Tous les paramètres pour reproductibilité
- `topics_over_time.csv` : Évolution temporelle des topics
- Modèle BERTopic complet

## Options Avancées (Optionnel)

### Désactiver certaines fonctionnalités

```bash
# Sans métriques de cohérence (plus rapide)
python topic_modeling.py --skip-coherence

# Sans analyse temporelle
python topic_modeling.py --skip-topics-over-time

# Les deux
python topic_modeling.py --skip-coherence --skip-topics-over-time
```

### Ajuster les paramètres de clustering

```bash
# Cibler environ 50 topics (défaut: 80)
python topic_modeling.py --desired-topics 50

# Réduire à un nombre fixe de topics après entraînement (fusion des similaires)
python topic_modeling.py --nr-topics 60

# Réduire les outliers (clusters plus gros) - valeurs par défaut optimisées
python topic_modeling.py --umap-n-neighbors 200 --hdbscan-min-samples 3

# Taille minimale des topics (défaut: 30)
python topic_modeling.py --min-topic-size 20
```

### Sauvegarder l'analyse temporelle ailleurs

```bash
python topic_modeling.py --save-topics-over-time results/temporal_analysis.csv
```

## Interprétation des Métriques de Cohérence

| Métrique | Plage | Bon score |
|----------|-------|-----------|
| **C_v** | 0-1 | ≥ 0.5 excellent, ≥ 0.4 acceptable |
| **NPMI** | -1 à 1 | Plus proche de 1 = mieux |
| **U_Mass** | Négatif | Plus proche de 0 = mieux |
| **Topic Diversity** | 0-1 | ≥ 0.8 = topics distincts |

## Dépendances

Les dépendances sont dans `requirements.txt`. Pour installer :

```bash
pip install -r requirements.txt
```

Dépendance importante : `gensim` (pour métriques de cohérence)

## Résolution de Problèmes

### Erreur de mémoire
→ Utilisez `--max-documents 5000` pour tester avec moins de docs

### Trop d'outliers (topic -1)
→ Les paramètres par défaut sont maintenant optimisés. Si besoin, augmentez encore `--umap-n-neighbors 250` et `--reduce-outliers-train 0.4`

### Trop de topics (fragmentation)
→ Utilisez `--nr-topics 60` pour fusionner les topics similaires après entraînement
→ Ou augmentez `--desired-topics 50` pour cibler moins de topics

### Topics avec mots anglais ou OCR
→ Les stopwords anglais sont maintenant filtrés par défaut. Pour ajouter des termes spécifiques, créez un fichier avec `--domain-stopwords-file`

### Topics trop larges/génériques
→ Réduisez `--desired-topics 100` ou `--min-topic-size 20`

### Gensim non disponible
→ Les métriques de cohérence seront sautées automatiquement (pas d'erreur)
