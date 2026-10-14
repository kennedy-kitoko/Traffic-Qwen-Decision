# comment l'architecture [https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) sera exelent pour ce role de traffic controle par son architecture que LLM(traffic-R1) ou jev ou laya
[[Résumé de notre recherche]]

Oui, l’architecture « Qwen + tête Clef » est particulièrement adaptée au contrôle du trafic. Mais il faut être précis : **Unsloth est le framework d’entraînement**. L’architecture décisionnelle est composée du backbone Qwen, d’une tête Clef et d’un espace d’actions typé.

```
État structuré du trafic
        ↓
Qwen3.5-0.8B spécialisé
        ↓
représentations de l’état et des 24 options
        ↓
tête Clef
        ↓
24 scores → softmax → 24 probabilités
        ↓
masque de sécurité
        ↓
phase + durée appliquées dans SUMO
```

## Pourquoi cette architecture correspond bien au trafic

Le contrôle d’une intersection n’est pas principalement un problème de rédaction. Le contrôleur doit choisir rapidement une action parmi un ensemble fini.

Dans notre cas :

\[ 4\ phases \times 6\ durées = 24\ actions \]

Traffic-Qwen V2 doit donc répondre à la question :

> Parmi les 24 combinaisons autorisées, laquelle présente les meilleures conséquences probables dans cet état ?

La documentation Unsloth explique que le modèle lit une seule fois l’état, les questions et les options. Une petite tête inspirée de Clef attribue ensuite un score à chaque option. Le modèle ne génère aucun texte pendant la décision. [Unsloth Decision Model documentation](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

### Une décision directe

Un LLM génératif produit normalement :

```
token 1 → token 2 → token 3 → ... → réponse finale
```

Traffic-Qwen produit directement :

```
24 logits → 24 probabilités → une action
```

Par exemple :

```
{
  "ETWT_15": 0.03,
  "ETWT_20": 0.06,
  "NTST_20": 0.09,
  "NLSL_15": 0.37,
  "NLSL_20": 0.29
}
```

Il n’est donc pas nécessaire :

- d’attendre la génération d’un raisonnement ;
- d’extraire une phase depuis une phrase ;
- de vérifier si le JSON généré est syntaxiquement valide ;
- de convertir une durée écrite en langage naturel ;
- de relancer le modèle à cause d’un format incorrect.

C’est un avantage important pour un système qui doit contrôler plusieurs intersections toutes les cinq secondes.

## Ce que fait exactement la tête Clef

La tête Clef ne demande pas au modèle de prédire le prochain mot. Elle examine les représentations calculées par Qwen pour l’état, la question et chaque option.

Elle produit un score :

\[ z_a=f_\theta(s,a) \]

pour chaque action \(a\), puis une distribution :

\[ P(a|s)=\operatorname{softmax}(z_a) \]

Le modèle apprend donc une fonction de classement :

```
état du trafic + action candidate
→ qualité probable de cette action
```

Cela correspond directement à notre dataset SUMO : chaque état contient les conséquences mesurées de plusieurs actions.

## Pourquoi les labels souples de SUMO conviennent à cette architecture

Supposons que SUMO mesure :

|Action|Retour|
|---|---|
|NLSL 15 s|0,2858|
|NLSL 20 s|0,2827|
|NTST 15 s|0,2100|
|ETWT 40 s|−0,0800|

La différence entre les deux premières actions est faible. Un label dur ferait croire au modèle que :

```
NLSL_15 = totalement correcte
NLSL_20 = totalement incorrecte
```

Une cible souple peut plutôt indiquer :

```
NLSL_15 : 0,34
NLSL_20 : 0,31
NTST_15 : 0,18
...
```

La tête Clef peut apprendre cette distribution complète. Elle apprend alors :

- quelle action est première ;
- quelles actions sont presque équivalentes ;
- quelles actions sont dangereuses ;
- le niveau d’incertitude de la décision.

C’est plus informatif qu’un simple label fourni par un expert.

## Pourquoi utiliser une tête conjointe à 24 actions

Traffic-Qwen V1 utilisait deux sorties :

```
phase : 4 probabilités
durée : 6 probabilités
```

Le problème est que ces sorties étaient partiellement indépendantes.

Le modèle pouvait produire :

```
phase la plus probable = NLSL
durée la plus probable = 20 s
```

alors que la meilleure combinaison simulée pouvait être :

```
NLSL 15 s
```

ou même :

```
ETWT 20 s
```

Avec la tête à 24 actions, la probabilité concerne directement une combinaison :

\[ P(\text{NLSL},15|s) \]

au lieu d’approximer :

\[ P(\text{NLSL}|s)\times P(15|s) \]

La dépendance entre la phase et la durée est donc préservée.

## Le rôle du masque de sécurité

La tête produit des probabilités sur les 24 actions, mais toutes ne sont pas toujours admissibles.

Par exemple :

- vert actuel inférieur à 15 s : aucune nouvelle action, commande `KEEP` ;
- vert actuel de 25 s : maintenir la même phase pendant 40 s dépasserait le plafond de 60 s ;
- une voie dépasse 60 s d’attente : sa phase devient prioritaire ;
- voie aval saturée : certaines actions peuvent être exclues.

Avant l’argmax, nous appliquons :

\[ z'_a = \begin{cases} z_a & \text{si } a \text{ est autorisée}\\ -\infty & \text{si } a \text{ est interdite} \end{cases} \]

puis :

\[ P(a|s,\text{autorisé})=\operatorname{softmax}(z'_a) \]

Traffic-Qwen choisit ainsi la meilleure action parmi les actions sûres.

L’intelligence apprise optimise le trafic ; les règles explicites garantissent les contraintes opérationnelles.

## Comparaison avec Traffic-R1

Traffic-R1 est un modèle de 3 milliards de paramètres entraîné avec exploration, renforcement et guidage expert. Son objectif inclut le raisonnement explicable, la communication entre intersections et la généralisation sans nouvel entraînement. Les auteurs annoncent également un déploiement opérationnel. [Article Traffic-R1](https://arxiv.org/abs/2508.02344)

|Élément|Traffic-R1|Traffic-Qwen Decision|
|---|---|---|
|Type|LLM avec raisonnement|modèle de décision|
|Taille|environ 3B|0,8B|
|Sortie|raisonnement et action|probabilités directes|
|Décodage|autorégressif|classement en une passe|
|Actions invalides|doivent être contrôlées après génération|masque déterministe|
|Explicabilité|forte|principalement probabiliste|
|Latence attendue|plus élevée|plus faible|
|Adaptation inattendue|potentiellement meilleure|dépend fortement des données|
|Déploiement local massif|plus exigeant|mieux adapté|
|Généralisation publiée|revendiquée et évaluée par Traffic-R1|pas encore démontrée pour Traffic-Qwen|

Traffic-R1 est plus approprié lorsqu’il faut :

- raisonner sur une situation nouvelle ;
- expliquer longuement une décision ;
- comprendre un incident décrit en langage naturel ;
- coordonner des situations complexes non présentes dans les données.

Traffic-Qwen est plus approprié lorsqu’il faut :

- prendre des milliers de décisions répétées ;
- garantir un format exact ;
- fonctionner avec peu de VRAM ;
- maîtriser la latence ;
- produire une distribution sur un ensemble fini ;
- exécuter localement une politique déjà spécialisée.

La meilleure architecture complète peut d’ailleurs utiliser les deux :

```
DeepSeek/Traffic-R1 comme expert hors ligne
→ SUMO valide les conséquences
→ Traffic-Qwen apprend les décisions
→ Traffic-Qwen contrôle en temps réel
```

Le raisonnement lourd sert à produire et analyser l’expérience. Le modèle décisionnel sert au déploiement.

## Comparaison avec Jev

Jev et Traffic-Qwen appartiennent à la même famille générale : les modèles System 1 qui prennent une décision typée sans générer un long texte.

Traffic-Qwen n’est donc pas architecturalement opposé à Jev. Il représente plutôt une manière de construire un modèle de type Jev avec :

- un backbone Qwen ouvert ;
- une tête Clef ;
- QLoRA ;
- un dataset trafic personnalisé ;
- un entraînement reproductible avec Unsloth ;
- un checkpoint local contrôlable.

L’avantage potentiel de Traffic-Qwen sur un Jev générique vient surtout de la spécialisation :

```
Jev générique
→ sait prendre des décisions générales

Traffic-Qwen
→ apprend la relation entre files, attentes, phases,
  durées et conséquences SUMO
```

Un Jev spécialisé sur exactement les mêmes données pourrait aussi devenir très performant. Il faudrait l’entraîner et le comparer dans les mêmes conditions avant de déclarer qu’une architecture est supérieure.

## Comparaison avec Laya

Laya est également un modèle non génératif. Son architecture utilise un encodeur bidirectionnel et des têtes de décision typées. Il renvoie des choix et des probabilités en une passe. Son dépôt annonce notamment une latence de 33 ms pour une question sur T4, avec de meilleures performances après spécialisation. [Dépôt Laya](https://github.com/kalufinnle/laya-decision-model)

Laya possède plusieurs avantages possibles :

- encodeur plus petit ;
- inférence très rapide ;
- architecture naturellement consacrée aux décisions ;
- coût mémoire inférieur ;
- bon support des questions typées.

Traffic-Qwen possède d’autres avantages possibles :

- représentations préentraînées plus riches de Qwen ;
- contexte plus long ;
- adaptation QLoRA facile ;
- écosystème Unsloth ;
- tête Clef entraînable avec le backbone ;
- possibilité future de réutiliser des capacités LLM plus générales.

L’effondrement observé avec Laya sans garde-fou ne prouve pas que Laya est une mauvaise architecture. Il montre surtout que le checkpoint générique ne connaissait pas suffisamment le domaine du trafic.

La comparaison scientifique correcte serait :

```
Traffic-Qwen entraîné sur V3
contre
Laya entraîné sur exactement V3
```

avec :

- mêmes états ;
- mêmes actions ;
- mêmes retours SUMO ;
- mêmes splits ;
- mêmes scénarios ;
- mêmes règles ;
- plusieurs graines.

## Pourquoi Qwen3.5-0.8B est un bon compromis

Qwen3.5-0.8B conserve une capacité de représentation importante tout en restant assez petit pour :

- la quantification 4 bits ;
- QLoRA sur une RTX 4060 8 Go ;
- l’inférence locale ;
- le traitement de plusieurs intersections ;
- l’expérimentation rapide ;
- la sauvegarde d’adapters légers.

La documentation Unsloth indique qu’un Qwen3.5-0.8B transformé en modèle de décision est passé de 36 % à 73 % sur leur benchmark `typed-decisions`, avec des gains importants sur BANKING77 et CLINC150. Cela démontre que la tête et la spécialisation peuvent fortement transformer le comportement du backbone, mais **ce benchmark n’est pas un benchmark de trafic**. [Résultats Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

Notre propre ablation Traffic-Qwen V1 est plus pertinente pour le trafic : le Qwen/Clef non entraîné restait bloqué sur une phase, tandis que les checkpoints spécialisés utilisaient les quatre phases. Cela constitue une preuve de spécialisation sur Jinan, mais pas encore une preuve de généralisation.

## Architecture recommandée pour Traffic-Qwen V2

```
État focal détaillé
+ résumé du réseau
+ phase actuelle
+ vert écoulé
+ attentes
+ files
+ arrivées
+ occupation aval
        ↓
Qwen3.5-0.8B 4-bit + LoRA
        ↓
tête Clef à 24 actions conjointes
        ↓
distribution apprise depuis les retours SUMO
        ↓
masque min 15 s / max 60 s / priorité attente
        ↓
argmax en évaluation
ou échantillonnage contrôlé en apprentissage
        ↓
phase + durée
        ↓
jaune 5 s si changement
        ↓
SUMO
```

Cette architecture est excellente pour Traffic-Qwen parce que le problème possède :

- un état structuré ;
- un petit espace d’actions discret ;
- des contraintes explicites ;
- une fréquence de décision élevée ;
- un besoin de probabilités ;
- un besoin de faible latence ;
- un simulateur capable de produire des labels fondés sur les conséquences.

Elle n’est pas automatiquement meilleure que Traffic-R1, Jev ou Laya. Sa force apparaît si nous réunissons quatre éléments :

1. un dataset SUMO diversifié et correctement séparé ;
2. des cibles conjointes phase-durée ;
3. une calibration fiable ;
4. des évaluations sur plusieurs épisodes, demandes, graines, villes et incidents.

La formulation scientifique juste est donc :

> Traffic-Qwen associe les représentations d’un petit modèle Qwen à une tête Clef non générative afin de transformer le contrôle des feux en classement probabiliste de 24 actions sûres. Cette architecture est mieux adaptée à l’inférence locale rapide qu’un LLM génératif, mais sa supériorité en matière de circulation dépendra de la qualité du dataset simulé et devra être démontrée face à Traffic-R1, Jev, Laya et MaxPressure sur des protocoles identiques.
