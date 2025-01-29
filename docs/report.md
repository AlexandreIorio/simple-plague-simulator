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

<!-- /code_chunk_output -->

<!-- pagebreak -->


## 0. Introduction

Ce laboratoire à pour but de d'accélerer une application demandant beaucoup de ressources en utilisant le parallélisme. Nous allons appliquer les concepts vus en cours pour accélérer une simulation de propagation de pandémie.

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

