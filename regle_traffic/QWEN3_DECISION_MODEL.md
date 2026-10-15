# Traffic-Qwen Decision Model

## Architecture décisionnelle pour le contrôle intelligent des feux de circulation

Traffic-Qwen transforme le contrôle d’un feu de circulation en un problème de **décision probabiliste discrète**.

Le contrôleur ne doit pas générer une longue réponse en langage naturel. À chaque état du trafic, il doit sélectionner rapidement une action parmi un ensemble fini de combinaisons **phase + durée**.

L’architecture retenue associe :

- un backbone **Qwen3.5-0.8B** ;
- une tête de décision **Clef** ;
- un espace de **24 actions conjointes** ;
- des cibles probabilistes construites à partir des conséquences mesurées dans **SUMO** ;
- un masque de sécurité déterministe ;
- un entraînement SFT, suivi à terme d’un apprentissage par renforcement dans le simulateur.

Unsloth est utilisé comme framework d’entraînement. Ce n’est pas Unsloth lui-même qui constitue l’architecture décisionnelle.

Documentation : [Train Your Own Decision Model with Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

---

## 1. Problème de décision

Traffic-Qwen contrôle quatre phases :

| ID | Phase |
| --- | --- |
| phase_0 | ETWT |
| phase_1 | NTST |
| phase_2 | ELWL |
| phase_3 | NLSL |

Chaque phase peut être associée à six durées :

```
15 s
20 s
25 s
30 s
35 s
40 s
```

L’espace d’action contient donc :

**4 phases × 6 durées = 24 actions**

Exemples :

```
phase_0_15
phase_0_20
phase_0_25
...
phase_3_35
phase_3_40
```

Le problème appris par le modèle est donc :

> Étant donné l’état actuel de l’intersection et du réseau, quelle combinaison phase–durée présente les meilleures conséquences probables parmi les actions actuellement admissibles ?

---

## 2. Architecture générale

```
État du trafic
      ↓
Qwen3.5-0.8B spécialisé
      ↓
Représentation de l’état et des actions candidates
      ↓
Tête Clef
      ↓
24 logits
      ↓
Masque des actions interdites
      ↓
Softmax sur les actions admissibles
      ↓
24 probabilités
      ↓
Sélection de l’action
      ↓
Phase + durée
      ↓
Contrôleur de sécurité
      ↓
SUMO / intersection réelle
```

Traffic-Qwen est donc utilisé comme **modèle de décision**, et non comme générateur de texte.

---

## 3. Pourquoi une sortie non générative

Un LLM génératif classique produit une réponse séquentiellement :

```
token 1 → token 2 → token 3 → ... → action
```

Traffic-Qwen produit directement :

```
état → 24 scores → 24 probabilités → action
```

Par exemple :

```json
{
  "phase_0_15": 0.03,
  "phase_0_20": 0.06,
  "phase_1_15": 0.09,
  "phase_2_15": 0.37,
  "phase_2_20": 0.29
}
```

Les valeurs ci-dessus sont seulement illustratives.

Cette approche évite, pendant le contrôle temps réel :

- la génération d’un raisonnement textuel ;
- l’extraction d’une phase depuis une phrase ;
- la génération et la validation d’un JSON ;
- l’interprétation d’une durée écrite en langage naturel ;
- les erreurs de format de sortie ;
- le coût d’un décodage autorégressif inutile pour une action discrète.

Le problème réel est une décision parmi 24 possibilités. L’architecture de sortie doit donc refléter directement cet espace d’action.

---

## 4. Rôle de la tête Clef

Pour un état de trafic **s**, la tête attribue un logit **z(a)** à chaque action candidate **a**.

```
z(a) = score(s, a)
```

Les logits sont ensuite transformés en probabilités :

```
P(a | s) = exp(z(a)) / Σ exp(z(j))
```

Le modèle apprend donc essentiellement :

```
état du trafic + action candidate
                ↓
score relatif de l’action
```

La sortie n’est pas un texte expliquant ce qu’il faudrait faire. C’est une distribution directement exploitable par le contrôleur.

---

## 5. Pourquoi utiliser 24 actions conjointes

Traffic-Qwen V1 séparait la décision en deux sorties :

```
phase  → 4 probabilités
durée  → 6 probabilités
```

Cette factorisation suppose implicitement qu’une bonne phase et une bonne durée peuvent être sélectionnées presque indépendamment.

Or les deux décisions sont liées.

Une durée de 40 secondes peut être excellente pour une phase fortement chargée et mauvaise pour une autre phase presque vide.

Avec deux distributions séparées, le système approxime une combinaison par :

```
P(phase, durée | état)
≈
P(phase | état) × P(durée | état)
```

Traffic-Qwen V2 apprend directement :

```
P(phase, durée | état)
```

Par exemple :

```
P(phase_2_15 | état)
P(phase_2_20 | état)
P(phase_2_25 | état)
...
```

La dépendance entre la phase et la durée est donc conservée.

C’est particulièrement important parce que le dataset SUMO évalue lui-même les combinaisons phase–durée comme des actions conjointes.

---

## 6. État fourni au modèle

Un exemple peut contenir notamment :

- l’intersection focale ;
- le temps de simulation ;
- la phase actuellement active ;
- le temps vert déjà écoulé ;
- la longueur des files ;
- le temps d’attente ;
- les véhicules en approche ;
- l’occupation des voies ;
- les informations aval disponibles ;
- un résumé des autres intersections du réseau ;
- les contraintes de sécurité actives ;
- la liste des actions admissibles et interdites.

Schématiquement :

```
État focal détaillé
+
Contexte du réseau
+
Phase actuelle
+
Temps vert écoulé
+
Files et attentes
+
Arrivées
+
Occupation
+
Contraintes opérationnelles
        ↓
Traffic-Qwen
```

Le modèle dispose ainsi d’un état structuré avant de classer les actions candidates.

---

## 7. Les labels viennent des conséquences SUMO

Le dataset V2 ne demande pas simplement à un LLM expert de fournir une action à imiter.

Pour chaque état sélectionné :

```
Épisode SUMO
    ↓
Trajectoire jusqu’au temps choisi
    ↓
Snapshot du réseau
    ↓
Masque de sécurité
    ↓
Proposition initiale du teacher
    ↓
Restauration du même snapshot
    ↓
Test des actions admissibles
    ↓
Mesure des conséquences pendant 120 s
    ↓
Calcul de la récompense
    ↓
Distribution probabiliste sur les actions
```

Chaque branche admissible part donc du même état initial et utilise le même protocole d’évaluation.

Le teacher aide à analyser l’état et les conséquences, mais la supervision principale est ancrée dans les résultats mesurés par le simulateur.

---

## 8. Fonction de récompense

Les conséquences d’une action sont évaluées à partir de plusieurs composantes du trafic, notamment :

- file moyenne ;
- temps d’attente ;
- débit ;
- changements de phase ;
- famine d’un mouvement ;
- spillback.

Schématiquement :

```
Reward =
  contribution_queue
+ contribution_waiting
+ contribution_throughput
+ contribution_switching
+ contribution_starvation
+ contribution_spillback
```

Les poids exacts sont définis par le protocole de génération du dataset.

Une action gagnante signifie donc :

> meilleure action parmi les actions évaluées dans cet état, avec cette fonction de récompense, cet horizon de simulation, ces contraintes et cette politique de continuation.

Cela ne signifie pas qu’elle est globalement optimale pour toutes les évolutions futures possibles du réseau.

---

## 9. Soft targets

Le dataset ne force pas systématiquement une cible one-hot.

Supposons que plusieurs actions donnent des résultats très proches :

| Action | Retour SUMO |
| --- | ---: |
| phase_3_15 | 0.2858 |
| phase_3_20 | 0.2827 |
| phase_1_15 | 0.2100 |
| phase_0_40 | -0.0800 |

Un label dur réduirait l’information à :

```
phase_3_15 = correcte
toutes les autres = incorrectes
```

Traffic-Qwen conserve plutôt une distribution probabiliste :

```json
{
  "phase_3_15": 0.34,
  "phase_3_20": 0.31,
  "phase_1_15": 0.18,
  "phase_0_40": 0.01
}
```

Ces nombres sont illustratifs.

Le modèle peut ainsi apprendre :

- quelle action est la meilleure ;
- quelles actions sont presque équivalentes ;
- quelles actions sont clairement mauvaises ;
- quand la décision est ambiguë ;
- quand une préférence est très forte.

Les soft targets sont particulièrement utiles lorsque plusieurs actions ont des conséquences similaires.

---

## 10. Masque de sécurité

L’intelligence apprise ne remplace pas les contraintes opérationnelles.

Traffic-Qwen peut attribuer un score à une action qui n’est pas admissible dans l’état courant. Le contrôleur applique donc un masque avant la sélection finale.

Pour chaque action :

```
si action autorisée :
    logit_final = logit_model

si action interdite :
    logit_final = -∞
```

Le softmax est ensuite calculé uniquement sur les actions admissibles.

Ainsi :

```
P(action interdite) = 0
```

Les probabilités restantes sont renormalisées.

---

## 11. Vert minimum et vert maximum

Le protocole impose :

```
vert minimum = 15 secondes
vert maximum = 60 secondes
```

Le temps considéré est le **temps vert continu effectif** de la phase.

### Avant 15 secondes

Un changement de phase est interdit.

```
elapsed_green < 15 s
→ changement de phase interdit
```

### Entre 15 et 60 secondes

Le modèle peut sélectionner une action admissible selon l’état du trafic et les autres contraintes.

```
15 s ≤ elapsed_green < 60 s
→ décision autorisée
```

### À 60 secondes

La phase ne peut plus être prolongée au-delà du maximum configuré.

```
elapsed_green = 60 s
→ extension de la même phase interdite
```

Le temps jaune n’est pas comptabilisé comme temps vert.

---

## 12. Une même phase signifie une extension

Lorsque le modèle choisit à nouveau la phase actuellement active, le compteur de vert ne doit pas être réinitialisé.

Exemple :

```
phase actuelle : phase_2
vert écoulé : 45 s
action demandée : phase_2_15
```

Résultat :

```
45 + 15 = 60 s
→ action admissible
```

Mais :

```
phase actuelle : phase_2
vert écoulé : 50 s
action demandée : phase_2_15
```

donnerait :

```
50 + 15 = 65 s
→ action interdite
```

Le maximum de 60 secondes porte donc sur le vert continu cumulé, pas uniquement sur la durée contenue dans l’action.

---

## 13. Sécurité dynamique pendant une branche SUMO

Une action peut être admissible au moment où elle commence puis devenir problématique pendant son évaluation.

Le simulateur continue donc à vérifier les règles pendant l’exécution.

Si une contrainte critique est déclenchée :

```
action en cours
      ↓
violation détectée
      ↓
interruption de la branche
      ↓
action marquée invalide
      ↓
probabilité cible = 0
```

Cela empêche le dataset d’enseigner comme souhaitable une action dont les conséquences violent les règles de sécurité.

---

## 14. Architecture d’apprentissage

La trajectoire prévue est :

```
Dataset SUMO contrefactuel
        ↓
SFT Traffic-Qwen
        ↓
Politique spécialisée initiale
        ↓
Évaluation dans SUMO
        ↓
Apprentissage par renforcement
        ↓
Politique améliorée
        ↓
Tests multi-scénarios
        ↓
Déploiement expérimental
```

### Étape 1 — SFT

Le SFT apprend au modèle les préférences issues des évaluations contrefactuelles SUMO.

L’objectif est de produire une politique initiale compétente et stable avant l’exploration RL.

### Étape 2 — RL

Après le SFT, Traffic-Qwen pourra interagir directement avec SUMO.

```
état
  ↓
Traffic-Qwen
  ↓
action
  ↓
SUMO
  ↓
nouvel état + récompense
  ↓
mise à jour de la politique
```

Le teacher n’est alors plus nécessaire pour chaque décision.

Le RL doit permettre au modèle d’améliorer sa politique à partir de ses propres trajectoires plutôt que de rester limité à l’imitation des décisions présentes dans le dataset SFT.

---

## 15. Pourquoi Qwen3.5-0.8B

Le backbone de 0,8 milliard de paramètres constitue un compromis entre capacité de représentation et coût de déploiement.

Les objectifs sont notamment :

- entraînement QLoRA sur du matériel accessible ;
- quantification ;
- faible consommation de VRAM ;
- inférence locale ;
- latence réduite ;
- possibilité de servir plusieurs intersections ;
- adaptation rapide à de nouveaux datasets ;
- conservation des représentations riches acquises pendant le préentraînement.

La documentation Unsloth rapporte des gains importants après transformation et spécialisation d’un petit Qwen en modèle de décision sur ses benchmarks de décisions typées.

Ces résultats démontrent la capacité de l’architecture à être spécialisée, mais ils ne constituent pas une preuve de performance en contrôle du trafic.

La performance trafic doit être mesurée directement dans SUMO puis, à terme, dans des environnements réels.

---

## 16. Comparaison avec un LLM génératif

| Propriété | LLM génératif de contrôle | Traffic-Qwen Decision |
| --- | --- | --- |
| Sortie | texte + action | action probabiliste |
| Décodage | autorégressif | décision en une passe |
| Format | doit être contrôlé | espace d’action fermé |
| Actions possibles | potentiellement ouvertes | 24 actions définies |
| Probabilités | pas toujours directement exploitables | natives |
| Masque d’actions | contrôle externe nécessaire | intégré au pipeline |
| Latence visée | plus élevée | plus faible |
| Explicabilité textuelle | forte | limitée |
| Déploiement local | plus coûteux | objectif principal |
| Contrôle répétitif | possible | architecture dédiée |

Traffic-Qwen sacrifie volontairement une partie de la flexibilité linguistique au profit d’un espace de décision fermé, contrôlable et rapide.

---

## 17. Comparaison avec Traffic-R1

Traffic-R1 explore l’utilisation d’un LLM spécialisé pour le contrôle des feux et met davantage l’accent sur le raisonnement, l’apprentissage par renforcement et les capacités générales du modèle.

Article : [Traffic-R1](https://arxiv.org/abs/2508.02344)

Traffic-Qwen poursuit un objectif différent :

```
Traffic-R1
→ modèle de contrôle fondé sur les capacités de raisonnement d’un LLM

Traffic-Qwen
→ modèle compact transformant directement un état en distribution d’actions
```

Traffic-R1 peut être intéressant lorsqu’un système doit raisonner explicitement sur des situations nouvelles ou complexes.

Traffic-Qwen vise surtout :

- les décisions répétitives à haute fréquence ;
- la faible latence ;
- l’inférence locale ;
- un format de sortie garanti ;
- un petit espace d’action ;
- un déploiement massif sur plusieurs intersections.

La supériorité de l’un ou de l’autre ne doit pas être supposée. Elle doit être mesurée sous un protocole commun.

---

## 18. Comparaison avec Jev

Jev et Traffic-Qwen partagent une idée importante : une tâche décisionnelle n’a pas nécessairement besoin d’une génération textuelle complète.

Traffic-Qwen applique cette philosophie spécifiquement au contrôle du trafic avec :

- un backbone Qwen ouvert ;
- une tête Clef ;
- 24 actions phase–durée ;
- un dataset SUMO spécialisé ;
- des soft targets ;
- un entraînement reproductible ;
- un checkpoint local contrôlable.

La comparaison scientifique pertinente n’est donc pas :

```
Qwen est meilleur que Jev
```

mais :

```
Traffic-Qwen spécialisé
vs
Jev spécialisé
```

avec les mêmes états, les mêmes actions, les mêmes scénarios et les mêmes métriques.

---

## 19. Comparaison avec Laya

Laya appartient également à la famille des modèles de décision non génératifs.

Dépôt : [Laya Decision Model](https://github.com/kalufinnle/laya-decision-model)

Ses avantages potentiels comprennent :

- petite architecture ;
- inférence rapide ;
- décisions typées ;
- faible coût mémoire.

Traffic-Qwen privilégie de son côté :

- les représentations du backbone Qwen ;
- la spécialisation QLoRA ;
- l’écosystème Unsloth ;
- la possibilité de conserver une capacité de représentation issue d’un LLM préentraîné.

Une comparaison correcte exige :

```
même dataset
mêmes splits
mêmes 24 actions
mêmes règles
mêmes épisodes SUMO
mêmes métriques
plusieurs seeds
```

Un modèle générique qui échoue sur le trafic ne démontre pas que son architecture est mauvaise. Il peut simplement ne pas avoir appris ce domaine.

---

## 20. Teacher LLM et Traffic-Qwen n’ont pas le même rôle

Le teacher utilisé pendant la construction du dataset et Traffic-Qwen ont des fonctions différentes.

### Teacher

Le teacher peut :

- analyser un état ;
- proposer une décision initiale ;
- examiner les conséquences simulées ;
- réviser son choix ;
- contribuer à l’analyse du dataset.

### SUMO

SUMO fournit les conséquences mesurées des actions candidates.

### Traffic-Qwen

Traffic-Qwen apprend ensuite une politique compacte à partir de ces expériences.

```
Teacher
   ↓
proposition et analyse
   ↓
SUMO
   ↓
conséquences mesurées
   ↓
soft targets
   ↓
Traffic-Qwen
   ↓
contrôle rapide
```

Traffic-Qwen ne cherche donc pas simplement à reproduire mot pour mot le teacher.

---

## 21. Architecture cible complète

```
Capteurs / SUMO
      ↓
Construction de l’état
      ↓
Intersection focale + contexte réseau
      ↓
Qwen3.5-0.8B
      ↓
Tête Clef
      ↓
24 logits phase–durée
      ↓
Masque de sécurité
      ↓
Softmax sur actions admissibles
      ↓
Distribution de probabilité
      ↓
Sélection
      ↓
Contrôleur déterministe
      ↓
Phase verte
      ↓
Transition de sécurité si nécessaire
      ↓
Nouvel état
```

Les responsabilités sont séparées :

```
Qwen + Clef
→ intelligence décisionnelle

Masque
→ admissibilité

Contrôleur de signal
→ exécution sûre

SUMO
→ environnement d’apprentissage et d’évaluation
```

---

## 22. Ce que Traffic-Qwen doit démontrer

L’architecture est prometteuse, mais la performance doit être démontrée expérimentalement.

Les principales hypothèses sont :

### H1 — Spécialisation

Le modèle spécialisé doit être meilleur que le checkpoint non spécialisé.

### H2 — Décision conjointe

La tête à 24 actions doit mieux préserver l’interaction phase–durée qu’une factorisation indépendante.

### H3 — Performance trafic

Traffic-Qwen doit améliorer les métriques de circulation par rapport aux baselines pertinentes.

### H4 — Faible latence

L’inférence doit être suffisamment rapide pour un contrôle local multi-intersections.

### H5 — Généralisation

Le modèle doit rester performant sur des épisodes, seeds, demandes et scénarios non vus pendant l’entraînement.

### H6 — Robustesse

Le système doit rester sûr sous congestion forte, changements de demande et situations atypiques.

### H7 — Apport du RL

Le RL doit améliorer la politique SFT sans provoquer d’effondrement, d’oscillations excessives ou de violations des contraintes.

---

## 23. Baselines d’évaluation

Traffic-Qwen devra être comparé à plusieurs familles de contrôleurs :

```
Fixed-Time
MaxPressure
Traffic-Qwen V1
Traffic-Qwen V2 SFT
Traffic-Qwen V2 SFT + RL
Jev spécialisé
Laya spécialisé
Traffic-R1 ou protocole comparable
```

Les comparaisons doivent utiliser autant que possible :

- les mêmes réseaux ;
- les mêmes fichiers de demande ;
- les mêmes seeds ;
- les mêmes horizons ;
- les mêmes règles de sécurité ;
- plusieurs épisodes ;
- des scénarios non vus.

Les métriques peuvent notamment inclure :

- temps de trajet ;
- temps d’attente ;
- longueur des files ;
- débit ;
- nombre de changements de phase ;
- starvation ;
- spillback ;
- latence d’inférence ;
- consommation mémoire.

---

## 24. Objectif de déploiement

L’objectif final n’est pas seulement d’obtenir de bonnes performances dans un benchmark.

Traffic-Qwen est conçu pour évoluer vers un contrôleur pouvant fonctionner localement à proximité des intersections.

```
Caméras / radar / boucles
        ↓
Extraction de l’état trafic
        ↓
Traffic-Qwen local
        ↓
Action phase–durée
        ↓
Contrôleur de sécurité indépendant
        ↓
Feu de circulation
```

Le modèle d’IA propose l’action.

Le contrôleur déterministe conserve l’autorité sur les contraintes critiques de sécurité.

Cette séparation est essentielle pour passer d’une expérience SUMO à une architecture de déploiement réel.

---

## 25. Principe central

Traffic-Qwen repose sur une idée simple :

> Le contrôle temps réel d’une intersection est avant tout un problème de décision sous contraintes, pas un problème de génération de texte.

Le projet combine donc :

```
représentations d’un petit Qwen
+
tête de décision non générative
+
24 actions conjointes phase–durée
+
conséquences mesurées dans SUMO
+
soft targets
+
contraintes de sécurité déterministes
+
SFT
+
RL
```

L’objectif scientifique n’est pas d’affirmer à l’avance que cette architecture est supérieure à Traffic-R1, Jev, Laya ou MaxPressure.

L’objectif est de vérifier expérimentalement si un petit modèle décisionnel spécialisé peut atteindre une qualité de contrôle élevée tout en offrant une **latence faible, un espace d’action strict, une inférence locale et une architecture adaptée au déploiement réel**.
