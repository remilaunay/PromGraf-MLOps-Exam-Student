# Bike Sharing MLOps Monitoring

API de prédiction du nombre de vélos loués (`cnt`) à partir du dataset [Bike Sharing UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), avec évaluation Evidently, métriques Prometheus, dashboards Grafana et alertes.

## Prérequis

- Docker Engine avec Docker Compose v2 (`docker compose`)
- GNU Make
- Un accès Internet au premier démarrage pour télécharger l’image Python et le dataset UCI

L’API télécharge le dataset puis entraîne une fois son `RandomForestRegressor` sur les données de janvier 2011 au démarrage du conteneur. Les rapports d’évaluation comparent les données reçues à ce jeu de référence.

## Démarrer les services

Depuis la racine du dépôt :

```bash
make
```

`make` lance `make all` : il construit l’image API et démarre l’API, Prometheus, Grafana et node-exporter en arrière-plan. Attendez que l’API réponde sur `/health` pendant son premier entraînement.

| Service | URL | Rôle |
| --- | --- | --- |
| API FastAPI | <http://localhost:8080> | Prédictions, évaluation et métriques |
| Documentation API | <http://localhost:8080/docs> | Schémas et endpoints interactifs |
| Prometheus | <http://localhost:19090> | Collecte des métriques et règles Prometheus |
| Grafana | <http://localhost:3000> | Dashboards et alertes |
| node-exporter | <http://localhost:9100/metrics> | Métriques CPU, mémoire et disque de l’hôte |

La connexion Grafana par défaut est `admin` / `admin`. Les variables `GRAFANA_ADMIN_USER` et `GRAFANA_ADMIN_PASSWORD` permettent de définir les identifiants. Prometheus est publié sur le port hôte `19090` par défaut pour éviter les conflits avec une autre instance utilisant `9090`; les ports peuvent être remplacés avec `API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` et `NODE_EXPORTER_PORT`.

## Commandes Makefile

| Commande | Action |
| --- | --- |
| `make` ou `make all` | Construit l’API et démarre les quatre services |
| `make stop` | Arrête et supprime les conteneurs du projet; les volumes de données persistent |
| `make evaluation` | Charge un échantillon UCI de février et l’envoie à `/evaluate` |
| `PERIOD=week2_february make evaluation` | Évalue une autre période disponible |
| `make traffic` | Évalue une période puis envoie 50 requêtes à `/predict` |
| `TRAFFIC_COUNT=10 make traffic` | Choisit le nombre de requêtes simulées |
| `make fire-alert` | Active l’indicateur de dérive pour vérifier l’alerte ML Grafana |
| `make status` | Affiche l’état des conteneurs Compose |
| `make logs` | Suit les journaux de tous les services |

`make fire-alert` met temporairement `evidently_data_drift_detected_status` à `1`. Grafana déclenche alors l’alerte **Bike Sharing input drift detected** et Prometheus l’alerte correspondante. `make evaluation` remplace cette valeur de démonstration par le résultat d’Evidently pour le nouveau lot.

## API et métriques

- `POST /predict` accepte les onze features du dataset ainsi que `dteday` et renvoie `predicted_count`.
- `POST /evaluate` accepte `{ "data": [...], "evaluation_period_name": "..." }`. Chaque ligne contient les features, `dteday` et la cible `cnt`.
- `GET /metrics` expose les compteurs de requêtes, les durées, RMSE, MAE, R², MAPE et le statut de dérive.
- `POST /trigger-drift` sert à démontrer l’alerte ML dans Grafana.

Les rapports [Evidently](https://docs.evidentlyai.com/) calculent RMSE, MAE et R² et détectent la dérive des features par comparaison avec janvier 2011. Le MAPE est calculé sur les lignes dont `cnt` est non nul, car le pourcentage d’erreur n’est pas défini pour une cible nulle.

## Dashboards et alertes

Grafana provisionne automatiquement la datasource Prometheus et trois dashboards depuis les fichiers versionnés :

- **API Performance** : débit des requêtes, latence P95 et taux d’erreurs HTTP 5xx.
- **Model Performance & Drift** : RMSE, MAE, R², MAPE et indicateur de dérive.
- **Infrastructure Overview** : utilisation CPU, mémoire et système de fichiers de l’hôte.

Le dashboard API complète les courbes avec le nombre de requêtes et de réponses 5xx sur cinq minutes; le taux 5xx reste affiché à 0 % lorsqu’aucune erreur n’est observée. Dans le dashboard modèle, les scores apparaissent après la première évaluation, le statut de dérive indique l’attente avant un premier lot et un panneau donne le nombre de lignes de la dernière évaluation.

L’alerte Grafana sur la dérive est provisionnée au démarrage et visible dans **Alerting**. Prometheus surveille également la disponibilité de l’API et la dérive des données.

Pour tester l’ensemble :

1. Lancez `make` et ouvrez Grafana.
2. Exécutez `make evaluation` pour alimenter les métriques du modèle.
3. Exécutez `make fire-alert` et vérifiez l’alerte dans Grafana et Prometheus.
4. Exécutez de nouveau `make evaluation` pour remettre à jour le statut de dérive à partir des données.
5. Lancez `make traffic` pour remplir les panels de performance API.

## Structure du projet

```text
.
├── README.md
├── README_exam.md
├── Makefile
├── docker-compose.yml
├── deployment/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── API-dashboard.json
│   │   │   ├── Infra-dashboard.json
│   │   │   └── Model-dashboard.json
│   │   └── provisioning/
│   │       ├── alerting/alert_rules.yaml
│   │       ├── dashboards/dashboards.yaml
│   │       └── datasources/datasources.yaml
│   └── prometheus/
│       ├── prometheus.yml
│       └── rules/alert_rules.yml
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   └── evaluation/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── run_evaluation.py
└── submission.tar.gz
```

## Démarche de développement, étape par étape

1. **Préparer l’API** : charger et traiter le dataset, entraîner le modèle sur janvier 2011, implémenter `/predict` et `/evaluate`, puis vérifier les réponses et `/metrics`.
2. **Instrumenter le modèle et l’API** : définir les métriques Prometheus, calculer les rapports Evidently et fournir `/trigger-drift` pour le scénario d’alerte.
3. **Assembler les services** : construire l’image API et configurer Compose, Prometheus, les règles d’alerte et node-exporter.
4. **Provisionner Grafana** : déclarer la datasource, les dashboards JSON et l’alerte ML dans les fichiers de provisioning versionnés.
5. **Automatiser les opérations** : piloter le démarrage, l’arrêt, l’évaluation, la simulation de trafic et le déclenchement d’alerte avec le Makefile.
6. **Vérifier chaque étape** : lancer `make`, `make evaluation`, `make traffic` et `make fire-alert`; contrôler les cibles Prometheus, les métriques et l’état de l’alerte Grafana.
7. **Versionner et livrer** : créer un commit pour chaque étape terminée, mettre à jour ce README et générer `submission.tar.gz` sans inclure le dépôt `.git` ni les caches locaux.
