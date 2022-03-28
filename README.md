## L2RPN 2022

Baselines for the 2022 edition of the L2RPN challenge


### Règles (pour chaque règles, on utilise obs.simulate):
- Si le maximum des rhos < ~90% :
  - Si les batteries sont chargées à moins
    de ~90% -> charger les batteries
  - Désactiver dispatch et curtailment (redevenir stable)
- Si max ~90% < rhos < ~95% :
  - Utiliser la batterie de gauche (à fond)
- Si max rhos > ~95% :
  - Utiliser les deux batteries (à fond)
  - Dispatch < 0 sur 0, 1 et > 0 sur 2