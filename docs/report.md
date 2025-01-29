<img src="logo.png" style="height:80px;">

# <center> Laboratoire n°05 {ignore=true}

# <center> Accélération d'une simulation de propagation de pandémie {ignore=true}

## <center>Département : TIC {ignore=true}

## <center>unité d’enseignement CNM {ignore=true}

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

Auteur: **André Costa & Alexandre Iorio**

Professeur: **Marina Zapater**

Assistant : **Mehdi Akeddar**

Salle de labo : **A09**

Date : **29.01.2025**



<!--pagebreak-->

## <center>Table des matières {ignore=true}

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=4 orderedList=false} -->

<!-- code_chunk_output -->

- [0. Introduction](#0-introduction)
- [1. Objectifs](#1-objectifs)
- [2. Description de l'application](#2-description-de-lapplication)
- [3. Analyse des performances de l'applicaton](#3-analyse-des-performances-de-lapplicaton)
- [4. Identification des goulets d'étranglement](#4-identification-des-goulets-détranglement)
  - [4.1 Détermination du nombre de cellules à vérifier](#41-détermination-du-nombre-de-cellules-à-vérifier)
    - [4.1.1 Cellule Healthy et Immune](#411-cellule-healthy-et-immune)
    - [4.1.2 Cellule Infected, Dead et Empty](#412-cellule-infected-dead-et-empty)
  - [4.2 Détermination du nombre de cellules à initialiser](#42-détermination-du-nombre-de-cellules-à-initialiser)
  - [4.3 Nombre total d'opérations](#43-nombre-total-dopérations)
  - [4.4 Solution pour accélérer l'application](#44-solution-pour-accélérer-lapplication)
    - [4.4.1 Parallélisation avec OpenMP](#441-parallélisation-avec-openmp)
    - [4.4.2 Parallélisation avec Cuda](#442-parallélisation-avec-cuda)
  - [4.5 Comparaison des performances](#45-comparaison-des-performances)
    - [4.5.1 Comparaisons des initialisations](#451-comparaisons-des-initialisations)
    - [4.5.2 Comparaisons des temps de simulation total de la pandémie](#452-comparaisons-des-temps-de-simulation-total-de-la-pandémie)
- [5. Identification des améliorations possibles](#5-identification-des-améliorations-possibles)
- [6. Conclusion](#6-conclusion)

<!-- /code_chunk_output -->

<!-- pagebreak -->


## 0. Introduction

Ce laboratoire a pour but d'accélérer une application demandant beaucoup de ressources en utilisant le parallélisme. Nous allons appliquer les concepts vus en cours pour accélérer une simulation de propagation de pandémie.
Toutes les simulations sont exécutées sur un serveur avec le matériel suivant :

```bash
cnm@cnm-desktop 
--------------- 
OS: Ubuntu 20.04.6 LTS aarch64 
Host: NVIDIA Orin Nano Developer Kit 
Kernel: 5.10.216-tegra 
Packages: 2138 (dpkg) 
Shell: bash 5.0.17 
Terminal: /dev/pts/2 
CPU: ARMv8 rev 1 (v8l) (6) @ 1.510GHz 
Memory: 1542MiB / 7451MiB 
```
et les caractéristiques `GPU` suivantes:

| Field                   | Value                 |
| ----------------------- | --------------------- |
| Device Name             | Orin                  |
| CUDA driver version     | 11.4                  |
| CUDA runtime version    | 11.4                  |
| CUDA Capability version | 8.7                   |
| Multiprocesors (MP)     | 8                     |
| CUDA cores/MP           | 128                   |
| Total CUDA cores        | 1024                  |
| GPU Max clock rate      | 624 MHz               |
| Global Memory           | 7451 MBytes           |
| Shared memory/block     | 49152 bytes           |
| Registers/block         | 65536                 |
| L2 Cache size           | 2097152 bytes         |
| Warp size               | 32                    |
| Max threads/block       | 1024                  |
| Max dim thread block    | x: 1024, y:1024, z:64 |



## 1. Objectifs

- Implémenter un programme de simulation de propagation de pandémie demandant beaucoup de ressources
- Analyser les performances de l'application
- Identifier les goulets d'étranglement de l'application
- Paralléliser l'application avec OpenMP
- Analyser les performances de l'application parallélisée avec OpenMP
- Paralléliser l'application avec Cuda
- Analyser les performances de l'application parallélisée avec Cuda
- Comparer les performances des trois versions de l'application
- Identifier les améliorations possibles

## 2. Description de l'application

L'application `simple-plague-simulation` disponible sur le dépôt `git` [simple-plague-simulator](https://github.com/AlexandreIorio/simple-plague-simulator.git) est une simulation de propagation de pandémie.
Son fonctionnement est simple : initialiser une `Grid 2D` de taille `N x N` avec un pourcentage d'occupation.
Chaque cellule de la grille peut être dans un des états suivants : `EMPTY`, `HEALTHY`, `INFECTED`, `DEAD` ou `IMMUNE`.
Nous décidons du nombre de personnes à infecter au début de la simulation et à chaque tour nous calculons les nouvelles infections en fonction des paramètres saisis. 
Une fois que plus personne n'est infecté, la simulation s'arrête et un rapport est généré.

**Exemple**
```bash
-----------------------------------
         Plague Simulator
-----------------------------------
Runtime : CPU
------------------------------------
Parameters
------------------------------------
Population                  : 50 %
World height                : 256
World Width                 : 256
World size                  : 65536
Proximity                   : 2
Infection duration          : 10 turns
Healthy infection probability:10 % 
Immune infection probability: 1 % 
Death probability           : 10 %
Initial infected            : 1
Population immunized        : 0%

-----------------------------------
         Initialisation
-----------------------------------
Initializing World ...
Initialization Duration: 0.00162655 s
------------------------------------
Initial world :
------------------------------------
Number of healthy people  : 32767
Number of infected people : 1
Number of immunized people: 0

------------------------------------
Simulation
------------------------------------
Simulation started
Round 0
Round 10
Round 20
Round 30
Round 40
Round 50
...
...
Round 510
Round 520
------------------------------------
Saving Timeline
Timeline created
Initialization took       : 0.00162655 s
Simulation took           : 1.17238 s
Total Time                : 1.174 s
Number of turns           : 523
Number of healthy people  : 429
Number of immunized people: 28281
Number of survivor        : 28710
Number of dead people     : 4058
```

## 3. Analyse des performances de l'applicaton
Dans un premier temps, nous avons tenté de mesurer les performances de l'application lancant simplement la simulation avec différentes tailles de grille.

Voici un graphique représentant le temps d'éxécution de la simulation en fonction de la taille de la grille:
![plot](performance_analysis_std.svg)

**Analyse**: Sur ce graphique, nous remarquons que le temps d'éxécution de la simulation augmente de manière linéaire avec un `R²` de `1` pour le temps de simulation, le temps par tour et le temps total. Quant à l'initialisation, le `R²` est de `0.995` ce qui signifie que le temps d'initialisation et moins prévisible mais reste relativement linéaire. Le compilateur semble optimiser le code pour les tailles de grille plus petites.

## 4. Identification des goulets d'étranglement

Pour déterminer le premier goulet d'étranglement, nous allons nous pencher sur le principe de fonctionnement de l'algorithm de propagation de pandémie.

En fonction de l'état de la cellule, le principe de verifier les voisins de chaque cellules, sur une distance representée par le paramêtre `proximity`, et de déterminer si la cellule doit changer d'état.

Cela augmente considérablement les opérations à effectuer.

Nous allons maintenant analyser le nombre d'opérations à effectuer pour chaque état de cellule.

### 4.1 Détermination du nombre de cellules à vérifier

#### 4.1.1 Cellule Healthy et Immune

Pour une cellule `HEALTHY` et `IMUNNE`, nous devons vérifier si un voisin est `INFECTED` et si la probabilité d'infection est respectée.

![proximity](proximity.png).

Nous pouvons déterminer le nombre de vérification avec les formules suivantes:

$$ Cells = (2 * proximity + 1)² $$. 

Maintenant, appliquons cette formule à une `Grid 2D` de taille `256x256` avec un `proximity` de `2`, avec un taux d'occupation de `50%` et 1 `INFECTED`. Analyse pour le premier tour:

$$ Grid_{size} = 256 * 256 = 65'536 $$
$$ Cells_{healthy} = Grid_{size} * 50\% = 32'768 $$
$$ Cells_{neighbours} = 5 * 5 = 25 $$
$$ Total_{NeighboursAnalysis} = Cells_{healthy} * Cells_{neighbours} = 819'200 $$
$$ Total_{cellsToCheck} = Total_{NeighboursAnalysis} + Grid_{size} = 884'736 $$
$$ Part_{NeighboursAnalysis} = \frac{Total_{NeighboursAnalysis}}{Total_{cellsToCheck}} = 92.5\% $$

En augmentant le `proximity` ou la taille de la `grid`, le nombre de cellules à vérifier augmente de manière exponentielle.

Analysons maintenant avec une `Grid 2D` de taille `4096 x 4096` toujours avec un `proximity` de `2` et un taux d'occupation de `50%` et 1 `INFECTED`. Analyse pour le premier tour:

$$ Grid_{size} = 4096 * 4096 = 16'777'216 $$
$$ Cells_{healthy} = Grid_{size} * 50\% = 8'388'608 $$
$$ Cells_{neighbours} = 5 * 5 = 25 $$
$$ Total_{NeighboursAnalysis} = Cells_{healthy} * Cells_{neighbours} = 209'715'200 $$
$$ Total_{cellsToCheck} = Total_{NeighboursAnalysis} + Grid_{size} = 226'492'416 $$
$$ Part_{NeighboursAnalysis} = \frac{Total_{NeighboursAnalysis}}{Total_{cellsToCheck}} = 92.5\% $$

De ce fait, nous pouvons dire que le nombre d'opérations, pour des paramètres `standard`, à une tendance à la baisse.

#### 4.1.2 Cellule Infected, Dead et Empty

Pour une cellule `INFECTED`, nous devons simplement vérifier si la durée d'infection est atteinte et si c'est le cas, la cellule devient `DEAD` ou `IMMUNE`.

Pour les cellules `DEAD` et `EMPTY`, nous devons simplement verifier leur status. 

Nous somme sur un algorithme de complexité `O(n)`.



Afin de déterminer le nombre de cellules à vérifier nous avons la formule suivante:

$$ Total_{cellsToCheck} = N * N $$

Maintenant, appliquons cette formule à une `Grid 2D` de différentes tailles.

$$ Grid_{size} = 256 * 256 = 65'536 $$
$$ Grid_{size} = 1024 * 1024 = 1'048'576 $$
$$ ... $$
$$ Grid_{size} = 4096 * 4096 = 16'777'216 $$

En augmentant la taille de la `grid`, le nombre de cellules à vérifier augmente linéairement par dimension et au carré pour la `grid`.

### 4.2 Détermination du nombre de cellules à initialiser

Afin de pouvoir jouer une simulation, il faut, dans un premier temps, initialiser la `grid` avec un taux d'occupation.

Afin de déterminer le nombre de cellules à initialiser nous avons la formule suivante:

$$ Cells_{toInitialize} = N * N $$

Maintenant, appliquons cette formule à une `Grid 2D` de différentes tailles.

$$ Grid_{size} = 256 * 256 = 65'536 $$
$$ Grid_{size} = 1024 * 1024 = 1'048'576 $$
$$ ... $$
$$ Grid_{size} = 4096 * 4096 = 16'777'216 $$

En augmentant la taille de la `grid`, le nombre de cellules à initialiser augmente linéairement par dimension et au carré pour la `grid`.

### 4.3 Nombre total d'opérations

Tout au long du déroulement du programme, la proportion de cellules va changer. 
En effet certaines cellule qui était `HEALTHY` vont probablement devenir `INFECTED`. Lorsque la cellule est `INFECTED`, ces voisins ne sont plus analysés. La durée de l'infection écoulée, la cellule va alors changer d'état pour être, soit `DEAD`, soit `IMMUNE`.

Dans le cas ou elle est `DEAD`, plus aucuns voisin ne sera analysé. 

En revanche, si la cellule devient `IMMUNE`, ses voisins seront a nouveau analysés. Il est alors possible qu'une cellule `IMMUNE` devienne `INFECTED` à nouveau.

Dans tout les cas, une opération permettant de vérifier l'état de la cellule est effectuée.

Analysons maintenant une pandémie complète avec une `Grid 2D` de taille `1024 x 1024` avec les paramètres suivants:
```bash
------------------------------------
Parameters
------------------------------------
Population Percentage          100%
World Width                    1024
World Height                   1024
World Size                     1048576
Proximity                      2
Infection Duration             5 rounds
Healthy Infection Probability  10%
Immune Infection Probability   1%
Death Probability              10%
Initial Infected               1
Initial Immune                 0
```
**Résultat**

```bash
Initialization took       : 1.56094 s
Simulation took           : 4.94875 s
Total Time                : 6.50969 s
Number of turns           : 783
Number of healthy people  : 11504
Number of immunized people: 902748
Number of survivor        : 914252
Number of dead people     : 134324
```

Une fois la simulation terminée, nous avons généré le graphique suivant:

![plot](pandemic_evolution.svg)

Faisons maintenant quelques hypothèses simple, pour déterminer le nombre total d'opérations à effectuer:

- Chaque cellule est analysée à chaque tour pour déterminer son état
$$ Total_{cellsToCheck} = 1024 * 1024 * 783_{tours} = 821'035'008 $$

- le grahique des personne `IMMUNE` peut être réduite à un triangle réctangle de base $783$ et de hauteur $\approx 900'000$:

$$ Total_{immuneToCheck} = 783 * 900'000 = 703'700'000 $$

- le grahique des personne `HEALTHY` est en forme de triangle isocel avec une base de $\approx 650_{round}$ pour une hauteur de $\approx 1'000'000_{healthyToCheck}$:

$$ Total_{healthyToCheck} = \frac{650 * 1'000'000}{2} = 650'000'000 $$

Le nombre d'opération peut être résumé de la manière suivante:

$$ Total_{operations} = Total_{cellsToCheck} + Total_{immuneToCheck} * 24 + Total_{healthyToCheck} * 24 = 33'309'835'008_{cellsChecked}$$

### 4.4 Solution pour accélérer l'application

Afin de paralléliser l'application, nous avons immaginé la solution suivante. 
Celle de passer une cellule ou un groupe de cellules de la `grid` à chaque thread.

#### 4.4.1 Parallélisation avec OpenMP

Notre première approche a été celle de paralléliser l'application avec `OpenMP`. 

De cette manière. nous pouvons remettre la charge de travail sur plusieurs coeurs de la `CPU`. 

Chaque thread va allors s'ocucper d'une partie de la `grid` pour les différentes tâches à effectuer.

##### 4.4.1.1 Initialisation


Pour l'initialisation, nous divisons la `grid` en `nb_chunk` parties de manière à initialiser sans conflit les cellules de la `grid`.

Voici in exemple pour `nb_chunk = 4`

![openmp](openmp_init.png)

De cette manière, les `threads` initialisent les cellules de la `grid` sans conflit.


##### 4.4.1.2 Simulation

//TODO

#### 4.4.2 Parallélisation avec Cuda

Notre deuxième approche a été celle de paralléliser l'application avec `Cuda`.
Cette approche différente de la première, nous permet de paralléliser l'application sur le `GPU`. Ainsi, comme nous avons un très grand nombre de `threads` sur le `GPU`, nous pouvons affecter un thread par cellule de la `grid`.

De cette manière, un thread fera un nombre de vérifications maximum de $ Cells_{neighbours} = (2 * proximity + 1)² $
 pour les cellules `HEALTHY` et `IMMUNE` et un nombre de vérification de `1` pour les cellules `INFECTED`, `DEAD` et `EMPTY`.

##### 4.4.2.1 Initialisation

L'initialisation avec `Cuda` est plus complexe que celle avec `OpenMP`. En effet, pour attribuer des cellules de manières aléatoires, nous devons utiliser une grille d'occupation. de cette manière, quand un thread veut ecrire sur une cellule, il peut verifier si la cellule est déjà occupée ou non, et si elle ne l'est pas, il peut l'occuper, et ce, de manière `atomique`.

##### 4.4.2.2 Simulation

Pour la simulation, la methode utilisée avec `Cuda` est bien plus simple, chaque thread va verifier les voisins de la cellule qu'il occupe et effectuer les actions necessaires, modifier un tableau temporaire et ensuite copier le tableau temporaire dans le tableau principal.

### 4.5 Comparaison des performances

Nous allons maintenant comparer les performances des différentes versions de l'application au différentes étapes de la simulation.

Pour cela, nous avons généré des graphiques représentant les performances de l'application pour une `grid` de `8192 x 8192`

![plot](performance_analysis_combined.svg)

Sur ce graphique, nous pouvons voir que l'implémentation `Cuda` est la plus performante devant `OpenMP` et la version `CPU` standard qui se trouve en dernière position.

Cependant, on remarque quelque chose d'intéressant au niveau de l'initialisation. L'implémentation `OpenMP` se trouve être la moins performante. Nous allons developper cette analyse dans la section suivante. 

#### 4.5.1 Comparaisons des initialisations

Afin de déterminer les divergences quant au résultat attendus, à savoir `Cuda` suivi de `OpenMP` et enfin la version `CPU` standard, nous avons généré un graphique zoomé sur les petites `grid`:

![plot](performance_analysis_zoomed.svg)

Comme nous l'avions vu précédemment, l'initialisation avec `OpenMP` est la moins performante. C'est chute de performance est dû au fait que `OpenMP` doit initialiser les `threads` et les `chunk` de la `grid` avant de pouvoir commencer l'initialisation. Probablement avec plus de coeurs, l'initialisation serait plus rapide. serait plus rapide.

On remarque bien que l'implémentaion `Cuda` est mauvaise pour les petites `grid`. La courbe de régression $y = mx + h$ possède une valeur de $h = 0.0404 [s]$, ce qui est énorme pour des petites `grid`. En effet, l'implémentation `Cuda` doit copier les données entre la `CPU` et le `GPU`. Pour des petites `grid`, le temps de copie est plus long que le temps d'éxécution

On remarque que l'implémentation `CPU` standard sera dépassée par `Cuda` dans les alentours de `2000 x 2000`. 

#### 4.5.2 Comparaisons des temps de simulation total de la pandémie

C'est sans surprise que l'implémentation `Cuda` est la plus performante, suivie de `OpenMP` et enfin la version `CPU` standard. 
cyb
En effet, une fois que les données sont chargées dans la `GPU` pour `Cuda`, les `threads` et les `chunks` définis pour `OpenMP`, la simulation peut alors performer.

L'ordre de grandeur des temps de simulation est de `1` pour `Cuda`, `3.5` pour `OpenMP` et `18.5` pour la version `CPU` standard.

Les graphiques de temps de simulation, temps par tour, de temps total réspecte les mêmes tendances.


## 5. Identification des améliorations possibles

Nous avons decidé, à chaque fois, de paralleliser la totalité de l'application avec `OpenMP` et `Cuda`. Cependant, après analyse, ont pourrait améliorer en sélectionnant la technologies 
en fonction de la taille de la `grid`.

Avec `Cuda` une optimisation qui accélérerait encore plus l'application ce serait d'optimiser le code pour diminurait le nombre d'accès à la mémoire globale de la `GPU` et donc augmenterait les performances.

## 6. Conclusion

//TODO