# Benchmark embeddings

Benchmark de 2 modeles d'embedding sur un echantillon de la base et des requetes metiers.

## Objectif
- Comparer les performances de 2 modeles d'embedding (multilingues)
- Construire des chunks, indexer avec FAISS, puis evaluer la recherche
- Mesures: recall@K et MRR

Modeles testes (par defaut dans `Bench_embedding.py`) :
- `paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-small`

## Prerequis
- Python 3.10+
- Acces a la base Postgres avec le schema `bench` (tables: corpus_docs, corpus_chunks, chunking_strategies, embedding_models, embedding_runs, faiss_indexes, queries, qrels, retrieval_results, metrics)
- FAISS (CPU par defaut)

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Par defaut, le script lit la connexion Postgres dans la variable d'environnement `APIMANGA_DSN`:

```bash
export APIMANGA_DSN='dbname=apimanga user=postgres password=postgres host=localhost port=5432'
```

Options utiles:
- `BENCH_OUT_DIR` (defaut: `./bench_out`)
- `BENCH_TOP_K` (defaut: `8`)
- `BENCH_DEBUG` (defaut: `1`)

### Option .env (local)
Un fichier `.env` n'est pas requis, mais tu peux l'utiliser en local pour centraliser la config.
Le repo fournit un modele dans `.env.example` (a copier en `.env` si besoin).
Les scripts chargent automatiquement `.env` via `python-dotenv`.

## Lancer le benchmark complet
Le script ci-dessous :
1) construit les chunks si besoin,
2) calcule les embeddings,
3) construit l'index FAISS,
4) evalue sur les requetes.

```bash
python3 Bench_embedding.py
```

Les artefacts FAISS sont ecrits dans `bench_out/faiss/<run_id>/` (index + meta).

## Evaluation uniquement (run_id existant)
Si tu veux re-evaluer sans recalculer les embeddings, utilise `Bench_only.py`.
Il charge les index FAISS existants references dans la DB (`bench.faiss_indexes`) :

```bash
python3 Bench_only.py
```

Par defaut, les `RUN_IDS` sont declares en haut de `Bench_only.py`. Mets-les a jour selon tes run_id.

## Resultats
Les metriques sont ecrites dans `bench.metrics`:
- `recall@K`
- `mrr`
- Pour `Bench_only.py`, les metriques sont prefixees par `entity_` (ex: `entity_recall@8`).

## Exemples de requetes metiers (benchmark)
- Humour / school / leger (Takagi): "Je cherche un manga dans le meme esprit que \"Quand Takagi me taquine\" : college, taquineries, humour leger, pas de violence."
- Romance / comedie (Kaguya): "Je veux une romance comique avec jeu psychologique entre deux lyceens, comme \"Kaguya-sama: Love Is War\"."
- Romance feel-good (Shikimori): "Une romance lycee feel-good, douce et drole, dans le style de \"Shikimori n'est pas juste mignonne\"."
- Shonen super-heros (My Hero Academia): "Un shonen d'action et de super-heros dans la veine de \"My Hero Academia\" (academie, progression, combats)."
- Shonen gangs / delinquants (Tokyo Revengers): "Un manga avec des gangs/affrontements, tension et drame, proche de \"Tokyo Revengers\"."
- Sport (Blue Lock): "Je cherche un manga de sport centre sur le football, competition, mental et depassement de soi, comme \"Blue Lock\"."
- Slice of life adulte / long cours (Space Brothers): "Un manga realiste et motivant sur une ambition de vie, carriere/projet au long cours, comme \"Space Brothers\"."
- Nocturne / surnaturel / romance (Call of the Night): "Ambiance nocturne, surnaturel leger et romance, dans le style de \"Call of the Night\"."
- Poetique / contemplatif (Du mouvement de la Terre): "Un recit profond et contemplatif, avec une dimension historique/philosophique, comme \"Du mouvement de la Terre\"."
- Imaginaire / onirique (Les Enfants de la Baleine): "Un univers onirique et melancolique, fantasy/aventure avec poesie, proche de \"Les Enfants de la Baleine\"."
- Competition / depassement (Chihayafuru): "Une serie centree sur la competition et le depassement de soi, avec un groupe de personnages, comme \"Chihayafuru\"."
- Shonen pont (Naruto + MHA): "Je veux un shonen proche de Naruto et My Hero Academia : progression, amities, rivalites, combats et valeurs positives."

## Notes
- Les embeddings sont normalises, et FAISS utilise `IndexFlatIP` (cosine sur vecteurs normalises).
- Le chunking utilise la strategie definie dans la DB par `CHUNKING_NAME`.
