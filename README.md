# Régression logistique from scratch

Prédiction de l'admission d'un étudiant à l'université à partir de ses notes à deux examens, avec une régression logistique implémentée entièrement en NumPy.

## Objectif

À partir d'un historique de 100 candidats (notes aux deux examens et décision d'admission), construire un classifieur capable d'estimer la probabilité qu'un nouveau candidat soit admis.

## Ce que contient le projet

- **Formulation probabiliste** du modèle : chaque étiquette suit une loi de Bernoulli de paramètre σ(xᵀθ), et les paramètres sont estimés par maximum de vraisemblance.
- **Implémentation vectorisée** de la sigmoïde, de la fonction de coût (entropie croisée binaire) et de son gradient, sous forme matricielle.
- **Tests unitaires**, dont une vérification du gradient analytique par différences finies.
- **Deux méthodes d'optimisation** : `scipy.optimize.minimize` (méthode TNC) et une descente de gradient codée à la main, avec standardisation des variables.
- **Visualisation** des données, de la frontière de décision et de la convergence du coût.
- **Comparaison avec scikit-learn** pour valider les résultats.

## Résultats

Les trois approches (SciPy, descente de gradient, scikit-learn) convergent vers les mêmes paramètres. Le modèle atteint environ **89 % de précision** sur les données d'entraînement avec une frontière de décision linéaire.

## Formules clés

En notant X la matrice des exemples (avec une colonne de 1 pour le biais) et h = σ(Xθ) :

```
J(θ)  = −(1/m) · [ yᵀ log(h) + (1 − y)ᵀ log(1 − h) ]
∇J(θ) =  (1/m) · Xᵀ (h − y)
```

## Structure du dépôt

```
.
├── data/
│   └── data1.txt                  # jeu de données (note 1, note 2, admis)
├── regression_logistique.ipynb    # notebook principal
├── requirements.txt
└── README.md
```

## Lancer le projet

```bash
git clone https://github.com/CtrlAltJoy/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch
pip install -r requirements.txt
jupyter notebook regression_logistique.ipynb
```

## Pistes d'amélioration

- Évaluer le modèle sur un jeu de test séparé ou par validation croisée
- Ajouter des variables polynomiales pour obtenir une frontière non linéaire
- Introduire une régularisation L2 pour limiter le surapprentissage

## Technologies

Python, NumPy, SciPy, Matplotlib, scikit-learn, Jupyter

## Crédits

Jeu de données issu du cours de Machine Learning d'Andrew Ng (Stanford / Coursera).
