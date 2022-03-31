# L2RPN_WCCI_baselines

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

## Performance de l'agent DoNothing:

The results for the DN agent are:
For chronics with id 2019-01-12
- cumulative reward: 4688.167480
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-13
- cumulative reward: 4685.237305
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-14
- cumulative reward: 3330.406250
- number of time steps completed: __219 / 288__

For chronics with id 2019-01-15
- cumulative reward: 1374.079468
- number of time steps completed: __89 / 288__

For chronics with id 2019-01-16
- cumulative reward: 3609.946533
- number of time steps completed: __235 / 288__

For chronics with id 2019-01-17
- cumulative reward: 3533.431396
- number of time steps completed: __233 / 288__

For chronics with id 2019-01-18
- cumulative reward: 1267.627197
- number of time steps completed: __85 / 288__

## Performance de l'agent ExpertAgent :

The results for the Expert3 agent are:
For chronics with id 2019-01-12
- cumulative reward: 4684.815918
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-13
- cumulative reward: 4682.062988
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-14
- cumulative reward: 3415.798584
- number of time steps completed: __225 / 288__

For chronics with id 2019-01-15
- cumulative reward: 4319.679199
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-16
- cumulative reward: 3620.062988
- number of time steps completed: __235 / 288__

For chronics with id 2019-01-17
- cumulative reward: 4320.351562
- number of time steps completed: __288 / 288__

For chronics with id 2019-01-18
- cumulative reward: 3503.213379
- number of time steps completed: 238 / 288

## Résumé :

Réussite :
- Nombre de pas de temps survécus bien supérieur (on passe deux scénarios de plus)

Mais règles incomplètes :
- Oscillations autour du seuil de 90%
- Les règles du redispatching marchent mal
- Situation où la centrale éolienne produit beaucoup non réglée