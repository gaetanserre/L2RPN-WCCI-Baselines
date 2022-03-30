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

|              | 01-12  | 01-13  | 01-14  | 01-15  | 01-16  | 01-17  | 01-18  | Mean  |
|-             |-       |-       |-       |-       |-       |-       |-       |-      |
| Do Nothing   | 288    | 288    | 219    | 89     | 235    | 233    | 85     | 205.2 |
| Expert Agent | 288    | 288    | 225    | 288    | 235    | 288    | 238    | 264.3 |

## Résumé :

Réussite :
- Nombre de pas de temps survécus bien supérieur (on passe deux scénarios de plus)

Mais règles incomplètes :
- Oscillations autour du seuil de 90%
- Les règles du redispatching marchent mal
- Situation où la centrale éolienne produit beaucoup non réglée