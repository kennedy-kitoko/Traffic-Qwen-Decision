L’objectif est de définir le protocole scientifique avant de demander à DeepSeek de produire des décisions. Ainsi, nous évitons de générer un dataset coûteux puis de modifier les règles après avoir vu les résultats.

Le principe central est le suivant :

```text
État SUMO réel
→ masque des actions autorisées
→ proposition structurée de DeepSeek
→ évaluation des actions autorisées dans SUMO
→ calcul des conséquences
→ validation et nettoyage
→ distribution cible sur les 24 actions
→ dataset compatible Traffic-Qwen V2
```

## 1. Espace de décision

Chaque décision est une combinaison entre une phase et une durée.

Les quatre phases sont :

| Phase | Mouvements servis |
|---|---|
| `ETWT` | Est et Ouest, mouvements directs |
| `NTST` | Nord et Sud, mouvements directs |
| `ELWL` | Est et Ouest, tourne-à-gauche |
| `NLSL` | Nord et Sud, tourne-à-gauche |

Les six durées sont :

```text
15, 20, 25, 30, 35 et 40 secondes
```

Cela produit 24 actions conjointes :

```text
ETWT_15, ETWT_20, ..., NLSL_35, NLSL_40
```

Pour Traffic-Qwen V2, il est préférable d’utiliser une sortie unique sur ces 24 actions. Cela conserve la relation entre la phase et sa durée.

Avec deux sorties indépendantes, le modèle pourrait sélectionner une phase probable et une durée probable dont la combinaison n’a pourtant jamais donné le meilleur résultat.

## 2. Vert minimal de 15 secondes

Une phase nouvellement activée doit rester verte pendant au moins 15 secondes.

Si :

\[
t_{\text{vert}} < 15
\]

alors le seul contrôle autorisé est :

```text
KEEP
```

Aucune nouvelle décision phase-durée ne devient un label d’entraînement à ce moment.

Cette règle évite :

- les changements trop rapides ;
- l’accumulation de transitions jaunes ;
- les oscillations entre deux phases ;
- les décisions qui ne laissent pas aux véhicules le temps de traverser.

`KEEP` reste une commande opérationnelle. Nous ne l’ajoutons pas comme vingt-cinquième action d’apprentissage pour le moment.

## 3. Vert continu maximal de 60 secondes

Une phase ne peut pas rester verte plus de 60 secondes consécutives.

Pour une action qui maintient la phase actuelle :

\[
t_{\text{écoulé}} + t_{\text{demandé}} \le 60
\]

Par exemple, si la phase actuelle est verte depuis 25 secondes :

- 15 secondes supplémentaires : autorisées ;
- 20 secondes : autorisées ;
- 25 secondes : autorisées ;
- 30 secondes : autorisées ;
- 35 secondes : autorisées, car le total vaut 60 ;
- 40 secondes : interdite, car le total atteindrait 65.

Les décisions sont appliquées par blocs de cinq secondes. Le garde-fou vérifie donc le compteur avant chaque bloc et ordonne le changement avant que le prochain bloc puisse dépasser 60 secondes.

Si le modèle veut encore conserver la même phase à la limite, le système choisit une autre phase présentant une demande. La demande est classée d’abord par file, puis par attente.

Cette règle évite qu’une phase monopolise indéfiniment l’intersection.

## 4. Priorité à une voie trop longtemps en attente

Le seuil actuel est :

```text
priority_wait_seconds = 60 secondes
minimum_queue_for_priority = 1 véhicule
```

Une voie devient prioritaire seulement si :

\[
\text{file} \ge 1
\]

et :

\[
\text{attente moyenne observée} \ge 60\,s
\]

Une voie vide ne devient donc jamais prioritaire uniquement à cause d’un ancien compteur d’attente.

La priorité concerne les phases concurrentes. Une voie appartenant à la phase actuelle est considérée comme étant en cours de service.

Si plusieurs voies concurrentes dépassent le seuil :

1. la plus grande attente moyenne gagne ;
2. en cas d’égalité, la plus grande file gagne ;
3. une règle déterministe sur l’ordre des phases résout la dernière égalité.

Lorsque la durée minimale de 15 secondes est déjà atteinte, la phase correspondant à la voie urgente devient obligatoire. Ses six durées restent candidates.

Cette règle empêche la famine d’une direction secondaire.

## 5. Ordre des règles

À chaque contrôle de cinq secondes, les règles sont appliquées dans cet ordre :

1. protéger le vert minimal de 15 secondes ;
2. rechercher une voie concurrente prioritaire ;
3. empêcher le dépassement des 60 secondes ;
4. appliquer la proposition du contrôleur si elle reste autorisée.

Cela signifie qu’une voie urgente ne peut pas interrompre une phase qui vient juste de commencer. Elle obtient la priorité dès que le minimum de sécurité est terminé.

## 6. Transition jaune

Lorsqu’une nouvelle phase est sélectionnée, JevLight/SUMO applique cinq secondes de jaune.

Une action demandant `NLSL` pendant 20 secondes depuis une autre phase consomme donc :

```text
5 secondes de jaune
+ 20 secondes de vert effectif NLSL
= 25 secondes de temps simulé
```

La durée choisie représente bien le temps vert. Elle n’inclut pas le jaune.

Lorsqu’on conserve la phase actuelle, aucune transition jaune supplémentaire n’est ajoutée.

## 7. État envoyé à DeepSeek

DeepSeek reçoit un état structuré, et non une image ou une description vague.

Pour l’intersection principale, l’état contient notamment :

- identifiant de l’intersection ;
- temps de simulation ;
- phase actuelle ;
- temps vert déjà écoulé ;
- file par mouvement ;
- attente moyenne observée par mouvement ;
- véhicules en approche ;
- informations disponibles sur les voisins ;
- phases disponibles ;
- durées disponibles ;
- règles de sécurité applicables.

Une synthèse du réseau contient notamment, pour chaque intersection :

- phase actuelle ;
- file totale ;
- nombre total de véhicules en approche ;
- attente maximale observée parmi ses mouvements.

DeepSeek connaît donc le contexte local et un résumé du réseau, mais il ne reçoit pas d’information future provenant de la simulation.

## 8. Rôle exact de DeepSeek

DeepSeek effectue deux opérations.

### Proposition initiale

Il examine l’état et propose :

```json
{
  "phase": "NLSL",
  "duration_seconds": 25
}
```

La réponse est contrôlée par Pydantic :

- aucune phase inconnue ;
- aucune durée différente des six durées autorisées ;
- aucun champ supplémentaire ;
- aucun nombre fourni sous une forme ambiguë ;
- aucune réponse libre utilisée comme action.

Si la proposition est interdite par le vert minimal, le plafond de 60 secondes ou la priorité d’une voie, elle est conservée pour l’audit, mais elle n’est pas exécutée comme action légale.

### Relecture des conséquences

Une fois les actions simulées, DeepSeek peut examiner leurs conséquences et expliquer pourquoi une autre action était meilleure.

Cette seconde réponse est aussi conservée comme métadonnée.

DeepSeek ne détermine cependant pas directement le label final. Le label vient des conséquences mesurées dans SUMO.

C’est la différence entre :

```text
action proposée par l’expert
```

et :

```text
action validée par la simulation
```

## 9. Évaluation contrefactuelle dans SUMO

Pour chaque état sélectionné :

1. SUMO atteint l’instant voulu ;
2. l’état du simulateur est sauvegardé ;
3. l’état Python de JevLight est également conservé ;
4. chaque action autorisée redémarre depuis exactement ce même état ;
5. la phase candidate est appliquée ;
6. la transition jaune est appliquée si nécessaire ;
7. la phase reste verte pendant la durée demandée ;
8. MaxPressure contrôle la suite de l’horizon ;
9. toutes les branches finissent après exactement 120 secondes.

MaxPressure sert de contrôleur commun après l’action candidate. Ainsi, une durée de 15 secondes ne bénéficie plus artificiellement d’un retour plus rapide à la trace historique de JevLight.

Toutes les actions sont comparées avec :

- le même état initial ;
- la même circulation ;
- la même graine ;
- le même horizon ;
- la même politique de continuation ;
- la même fonction de récompense.

## 10. Actions interdites et actions interrompues

Les 24 actions restent l’espace de sortie du modèle, mais elles ne sont pas toutes autorisées dans chaque état.

Le dataset contiendra donc un masque :

```json
{
  "eligible_actions": ["phase_3_15", "phase_3_20", "..."],
  "forbidden_actions": ["phase_0_15", "phase_1_15", "..."]
}
```

Une action interdite recevra une probabilité cible nulle.

Une situation plus délicate peut apparaître pendant l’exécution : une nouvelle voie peut dépasser le seuil d’attente alors que l’action candidate est encore en cours.

Dans ce cas, la règle de priorité interrompt l’action. Cette branche est rejetée pour l’apprentissage, car il serait faux d’affirmer que la phase et la durée demandées ont été entièrement exécutées.

## 11. Récompense mesurée

Chaque action produit plusieurs mesures séparées :

- file moyenne ;
- attente accumulée ;
- véhicules arrivés à destination ;
- changements de phase ;
- famine ;
- occupation aval et spillback.

La récompense actuelle est configurable et prend la forme :

\[
R =
-0.30Q
-0.25W
+0.20T
-0.05S
-0.10F
-0.10B
\]

avec :

- \(Q\) : file normalisée ;
- \(W\) : attente normalisée ;
- \(T\) : débit ou véhicules arrivés ;
- \(S\) : changements de phase ;
- \(F\) : famine ;
- \(B\) : spillback.

Toutes les composantes brutes sont conservées. Le dataset ne conservera donc pas seulement un score final impossible à vérifier.

La normalisation borne chaque composante afin d’éviter qu’une mesure numériquement très grande écrase automatiquement toutes les autres.

## 12. Construction de la cible d’apprentissage

Nous ne prendrons pas automatiquement la meilleure action comme label dur.

Si deux actions ont des résultats très proches, par exemple :

```text
meilleure action : 0,285776
deuxième action : 0,282701
écart : 0,003075
```

il serait excessif de déclarer que la première est absolument correcte et toutes les autres incorrectes.

Les retours mesurés seront transformés en probabilités :

\[
P(a|s)=
\frac{\exp((G(a)-G_{\max})/\tau)}
{\sum_{a'}\exp((G(a')-G_{\max})/\tau)}
\]

La température \(\tau\) détermine si la distribution est :

- très concentrée sur la meilleure action ;
- modérément souple ;
- presque uniforme.

La température sera sélectionnée uniquement à partir des épisodes d’entraînement et de validation. Le test final ne servira jamais à choisir cette valeur.

Pour une action interdite :

\[
P(a|s)=0
\]

La distribution est ensuite renormalisée uniquement sur les actions admissibles.

## 13. Composition d’une ligne du dataset

Une ligne V3 devrait être organisée comme ceci :

```json
{
  "state": {
    "simulation_time_seconds": 301,
    "focal_intersection_id": "intersection_1_1",
    "intersection": {
      "current_phase": "ETWT",
      "current_phase_elapsed_green_s": 25,
      "lanes": {}
    },
    "network_summary": {},
    "safety_constraints": {
      "minimum_green_seconds": 15,
      "maximum_continuous_green_seconds": 60,
      "priority_wait_seconds": 60,
      "minimum_queue_for_priority": 1
    }
  },
  "questions": {
    "traffic_action": {
      "type": "choice",
      "criteria": {
        "phase_0_15": "Activate ETWT for 15 seconds",
        "...": "...",
        "phase_3_40": "Activate NLSL for 40 seconds"
      }
    }
  },
  "gold": {
    "traffic_action": {
      "label": "phase_3_15",
      "choice": "phase_3_15",
      "probabilities": {
        "phase_0_15": 0.0,
        "...": 0.0,
        "phase_3_15": 0.31,
        "phase_3_20": 0.27
      }
    }
  },
  "metadata": {
    "teacher_proposal": {},
    "teacher_review": {},
    "eligible_actions": [],
    "forbidden_actions": [],
    "raw_returns": {},
    "reward_components": {},
    "seed": 3407,
    "episode_id": "...",
    "traffic_file": "...",
    "snapshot_sha256": "..."
  }
}
```

Le champ principal appris est `traffic_action`, avec les 24 combinaisons conjointes.

## 14. Séparation train, validation et test

Les états ne seront pas mélangés aléatoirement.

Les épisodes complets seront affectés à un split avant la sélection des états :

```text
épisodes train
épisodes validation
épisodes test
```

Aucun état, snapshot ou fragment temporel d’un épisode test ne pourra servir :

- à entraîner le modèle ;
- à choisir la température ;
- à régler la récompense ;
- à choisir les seuils ;
- à sélectionner les hyperparamètres.

Cette séparation doit avoir lieu avant les appels DeepSeek et avant les évaluations contrefactuelles.

## 15. Nettoyage du dataset

Une comparaison sera rejetée si au moins une condition suivante apparaît :

- snapshot SUMO non fidèle ;
- différence entre l’état original et l’état restauré ;
- timer de vert absent ;
- phase ou durée inconnue ;
- réponse DeepSeek invalide ;
- erreur dans une branche SUMO ;
- horizon différent entre les actions ;
- graine différente ;
- politique de continuation différente ;
- configuration de récompense différente ;
- valeur `NaN` ou infinie ;
- composante de récompense manquante ;
- score impossible à recalculer ;
- action préemptée pendant son engagement ;
- chevauchement entre épisodes train et test ;
- doublon d’état ;
- fichier de trafic différent de celui déclaré ;
- action interdite ayant reçu une probabilité positive.

Les comparaisons rejetées seront conservées séparément pour l’audit. Elles ne deviendront pas silencieusement des exemples d’entraînement.

## 16. Détection des doublons

Chaque état recevra un hash normalisé construit à partir des éléments significatifs :

- intersection ;
- phase ;
- temps vert écoulé ;
- files ;
- attentes ;
- arrivées ;
- contexte réseau ;
- scénario ;
- temps de simulation.

Deux états identiques ou presque identiques provenant d’instants voisins ne devront pas se retrouver dans deux splits différents.

Une distance temporelle minimale sera également appliquée pour éviter de collecter une longue série d’états presque identiques toutes les cinq secondes.

## 17. Tests avant la grande génération

Les validations sont organisées par niveaux.

### Tests unitaires

Ils couvrent notamment :

- vert minimal ;
- plafond de 60 secondes ;
- voie vide sans priorité ;
- voie réellement demandée et trop longtemps en attente ;
- choix de la plus grande attente ;
- masque des actions ;
- transition jaune ;
- récompense ;
- famine ;
- spillback ;
- validation Pydantic ;
- probabilités normalisées.

Les 17 tests ciblés actuels passent.

### Smoke test SUMO réel

Un état Jinan a été restauré et simulé pendant 120 secondes avec les nouvelles règles :

- snapshot fidèle ;
- aucune erreur ;
- plafond vérifié à chaque pas ;
- transition jaune conservée ;
- 120 secondes de mesures obtenues.

### Petit pilote

Avant une collecte importante, il faudra générer quelques états provenant de plusieurs épisodes et vérifier :

- nombre d’actions légales par état ;
- diversité des phases gagnantes ;
- diversité des durées gagnantes ;
- différence entre les actions ;
- fréquence de la priorité ;
- fréquence du plafond de 60 secondes ;
- entropie des distributions ;
- écarts entre meilleure et deuxième action ;
- coût API ;
- coût SUMO ;
- taux de rejet.

### Validation Unsloth/Clef

Un petit échantillon devra ensuite être chargé par :

```python
FastDecisionModel.build_dataset(...)
```

Nous vérifierons :

- aucune ligne ignorée ;
- aucune troncature ;
- 24 probabilités ;
- somme proche de 1 ;
- actions interdites à zéro ;
- forward pass ;
- backward pass ;
- sauvegarde et rechargement.

## 18. Pourquoi définir tout cela avant DeepSeek ?

Il y a plusieurs raisons essentielles.

Premièrement, DeepSeek ne doit pas décider lui-même ce qui est physiquement ou opérationnellement autorisé. Les contraintes de sécurité viennent du système de contrôle.

Deuxièmement, le coût des appels API et des simulations contrefactuelles est réel. Détecter après plusieurs centaines d’appels que le timer manque ou que les actions sont mal étiquetées gaspillerait toute la collecte.

Troisièmement, définir la récompense, les seuils et les splits après avoir vu les résultats introduirait un biais scientifique. On pourrait inconsciemment choisir les règles qui rendent les résultats plus favorables.

Quatrièmement, le LLM peut produire un raisonnement convaincant et pourtant choisir une mauvaise action. SUMO doit rester l’arbitre des conséquences.

Cinquièmement, les premières expériences ont montré que les meilleures actions sont parfois séparées par une différence très faible. Les probabilités souples sont donc plus honnêtes que des labels artificiellement certains.

Enfin, les durées et les phases sont liées. Une représentation conjointe à 24 actions évite de recombiner deux décisions indépendantes qui n’ont jamais été validées ensemble.

## État actuel réel

Les règles de sécurité et leur test SUMO sont prêts. Les anciens exemples et les résultats Traffic-Qwen V1 n’ont pas été modifiés.

Il reste à terminer le constructeur spécifique du dataset V3 avec :

- masque variable des actions légales ;
- probabilités nulles pour les actions interdites ;
- normalisation sur les actions autorisées ;
- conservation des retours bruts ;
- rejet explicite des branches préemptées.

Aucun nouvel appel DeepSeek ne devrait être lancé avant que ce constructeur V3 et ses tests soient terminés. C’est cette dernière étape qui garantira que les prochaines données générées seront directement exploitables pour entraîner Traffic-Qwen V2.
