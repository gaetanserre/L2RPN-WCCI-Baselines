# L2RPN WCCI Baselines

Baselines for the 2022 edition of the L2RPN challenge

## Règles:

Règles établies suite à plusieurs essais sur grid2game

- Si le maximum des rhos < ~90% :
  - Si les batteries sont chargées à moins
    de ~95% -> charger les batteries
  - Remettre à 0 les redispatching
- Si max ~90% < rhos < ~95% :
  - Utiliser la batterie de gauche en mode production
- Si max rhos > ~95% :
  - Utiliser les deux batteries en mode production
  - Redispatching < 0 sur générateurs 0 et 5 et redispatching > 0 sur générateur 1

On simule l'action avant de l'appliquer, en cas de prévision d'un game over, on ne fait rien.

__Tableau des nombres de pas de temps survécus__ : (sur un total de 288)

| Agent\Scenario | 01-12  | 01-13  | 01-14  | 01-15  | 01-16  | 01-17  | 01-18  | Mean  |
|-               |-       |-       |-       |-       |-       |-       |-       |-      |
| Do Nothing     | 288    | 288    | 219    | 89     | 235    | 233    | 85     | 205.2 |
| Expert Agent   | 288    | 288    | 225    | 288    | 235    | 288    | 238    | 264.3 |

## Résumé :

__Réussite__ :
- Nombre de pas de temps survécus bien supérieur (on passe deux scénarios de plus)

__Mais règles incomplètes__ :
- Oscillations autour du seuil de 90%
- Les règles du redispatching marchent mal
- Situation où la centrale éolienne produit beaucoup non réglée : il faudrait écrêter

__Travail investi__ :
- Temps consacré :
  - Gaëtan : un après-midi
  - Eva : l'équivalent de deux jours
- 3 versions d'algorithme testées
- Une dizaine de configurations de seuils essayées

On réussit seulement 2 scénarios en plus

__Limites__ :
- La détermination des bons seuils est très chronophage
- Gestion temporelle non prise en compte : si on maintient la batterie en charge pendant plusieurs pas de temps quand il y a un problème (afin d'éviter les oscillations), gestion de sa recharge compliquée
- Nous rencontrons des difficultés sur le cas 14 noeuds et deux batteries. Il y aura encore plus de difficultés sur le cas 118 noeuds avec encore plus de batteries
- Cette approche n'est pas robuste à un changement de topologie, qu'il soit décidé (maintenance) ou subi (N-1)
