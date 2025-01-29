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

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=3 orderedList=false} -->

<!-- code_chunk_output -->

- [0. Introduction](#0-introduction)
- [1. Objectifs](#1-objectifs)
- [2. Description de l'application](#2-description-de-lapplication)
- [3. Analyse des performances de l'applicaton](#3-analyse-des-performances-de-lapplicaton)
- [4. Identification des goulets d'étranglement](#4-identification-des-goulets-détranglement)
  - [4.1 Détermination du nombre de cellules à vérifier](#41-détermination-du-nombre-de-cellules-à-vérifier)
  - [4.2 Détermination du nombre de cellules à initialiser](#42-détermination-du-nombre-de-cellules-à-initialiser)

<!-- /code_chunk_output -->

<!-- pagebreak -->


## 0. Introduction

Ce laboratoire à pour but de d'accélerer une application demandant beaucoup de ressources en utilisant le parallélisme. Nous allons appliquer les concepts vus en cours pour accélérer une simulation de propagation de pandémie.
Toute les simulations sont éxécutées sur un serveur sur le matériel suivant:

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

- Implémenter une programme de simulation de propagation de pandémie demandant beaucoup de ressources
- Analiser les performances de l'application
- Identifier les goulets d'étranglement de l'application
- Paralléliser l'application avec `OpenMP`
- Analyser les performances de l'application parallélisée avec `OpenMP`
- Paralléliser l'application avec `Cuda`
- Analyser les performances de l'application parallélisée avec `Cuda`
- Comparer les performances des trois versions de l'application
- Identifier les améliorations possibles

## 2. Description de l'application

L'application `simple-plague-simulation` disponible sur le dépôt `git` [simple-plague-simulator](https://github.com/AlexandreIorio/simple-plague-simulator.git) est une simulation de propagation de pandémie.
Son fonctionnement est simple, initialiser une `Grid 2D` de taille `N x N` avec un pourcentage d'occupation.
Chaque cellule de la grille peut être dans un des états suivants: `EMPTY`, `HEALTY`, `INFECTED`, `DEAD` ou `IMMUNE`.
Nous décidons du nombre de personne à infécter au début de la simulation et chaque tour nous calculons les nouvelles infections en fonction des paramêtres saisis. 
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

**Analyse**

Sur ce graphique, nous remarquons que le temps d'éxécution de la simulation augmente de manière linéaire avec un `R²` de `1` pour le temps de simulation, le temps par tour et le temps total. Quant à l'initialisation, le `R²` est de `0.995` ce qui signifie que le temps d'initialisation et moins prévisible mais reste relativement linéaire. Le compilateur semble optimiser le code pour les tailles de grille plus petites.

## 4. Identification des goulets d'étranglement
En fonction de l'état de la cellule, le principe de l'application est de verifier les voisins de chaque avec un `radius` qui est representé par le paramêtre `proximity`. 
C'est à dire, pour un paramètre proximity de `2`, nous devons vérifier, dans certains cas, les voisins dans un carré de `5x5` autour de la cellule.

### 4.1 Détermination du nombre de cellules à vérifier

#### 4.1.1 Cellule Healthy et Immune

Pour une cellule `HEALTHY` et `IMUNNE`, nous devons vérifier si un voisin est `INFECTED` et si la probabilité d'infection est respectée.

![proximity](proximity.png).

Afin de déterminer le nombre de cellules à vérifier nous avons la formule suivante:

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

Tout au long du dérouement du programme, la proportion de cellules va changer. En effet certaines cellule qui était `HEALTHY` vont devenir `INFECTED` et donc ne plus être analysées, puis peut-être devenir `IMMUNE` et continuer à être analysées, ou alors devenir `DEAD` et ne plus être analysées, jusqu'à la fin de la simulation.

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






