# Prédiction conforme : notes de cours et indices du TP

Résumé du CM de Joseba Dalmau (`docs/conformal/[Lecture] Conformal Prediction.pdf`, version d'octobre 2026, 75 slides). Le PDF à la racine du repo est l'ancienne version de 2024.

## 1. L'idée
La CP prend un modèle déjà entraîné et renvoie un **ensemble** (intervalle ou classes) qui contient la vraie réponse avec probabilité ≥ 1−α :

    P(Y_test ∈ Ĉ_α(X_test)) ≥ 1−α

Pourquoi : les réseaux de neurones sont mal calibrés (un P = 0,998 n'est pas juste 99,8 % du temps). Bayésien, ensembles et dropout ne sont pas post-hoc et n'ont pas de garantie.
Analogie : un météorologue regarde ses erreurs passées sur des jours que le modèle n'a pas vus. 90 % du temps, il s'est trompé de moins de 3 °C, donc il annonce 18 ± 3 °C.

## 2. Split conformal (régression)
1. Entraîner f̂ sur D_train. Garder D_calib à part (n points jamais vus par le modèle).
2. Scores S_i = |y_i − f̂(x_i)| sur D_calib.
3. δ_α = le ⌈(n+1)(1−α)⌉-ième plus petit score (le quantile ⌈(n+1)(1−α)⌉/n).
4. Prédire Ĉ_α(x) = [f̂(x) − δ_α, f̂(x) + δ_α].

Exemple : n = 9, α = 0,2. Les scores triés sont 0,1 0,3 0,5 0,8 0,9 1,2 1,5 2,0 3,1. On calcule ⌈10×0,8⌉ = 8, donc δ = 2,0. Si f̂(x) = 10, on prédit [8 ; 12].
Contrainte : si α < 1/(n+1), alors δ = +∞ et Ĉ = ℝ.

**Théorème (Vovk 1999)** : si les scores (calibration + test) sont échangeables, la garantie tient. i.i.d. implique échangeable. Exemples échangeables non i.i.d. : Z_i/Σ|Z_j|, ou des tirages sans remise.
**Preuve** : le rang de S_test parmi les n+1 scores est uniforme, donc P(rang ≤ K) = K/(n+1). On prend K* = ⌈(n+1)(1−α)⌉. Or rang ≤ K* ⇔ S_test ≤ δ_α ⇔ Y ∈ Ĉ.
**Avantages** : post-hoc, distribution-free, hypothèses minimales, garantie en échantillon fini. Si f̂ est mauvais, la garantie tient quand même, mais l'intervalle est énorme.
**Forme générale** : score s(ŷ, y), q̂ = quantile corrigé, C(x) = {y : s(ŷ, y) ≤ q̂}. C'est la classe `SplitConformal` du TP.

## 3. Limites et remèdes
| Limite | Remède |
|---|---|
| 1. Garantie marginale sur la calibration (pour un jeu fixé, la couverture suit une loi Beta(n+1−l, l), l = ⌊(n+1)α⌋) | n grand (~1 000 points) |
| 2. Non conditionnelle (P(Y∈C \| X=x) ≥ 1−α pour tout x est impossible, Vovk 2012) | Group-balanced CP (un split par groupe), Gibbs et al. 2024 |
| 3. Besoin de données de calibration | Jackknife+ / CV+ (K modèles, garantie ≥ 1−2α, ≈ 1−α en pratique) |
| 4. Hétéroscédasticité (largeur constante) | Locally adaptive (S = \|f̂−y\|/σ̂(x)), CQR |

**CQR** : S = max(q̂_{α/2}(x) − y, y − q̂_{1−α/2}(x)), qui est négatif si y est dans la bande. On prédit C(x) = [q̂_{α/2}(x) − δ, q̂_{1−α/2}(x) + δ]. Si δ < 0, on rétrécit la bande.
**Évaluation** : la couverture (≈ 1−α) et la taille moyenne. Un ensemble plus petit n'est pas toujours meilleur : il vaut mieux qu'il s'adapte au bruit.

## 4. Classification
| Méthode | Score | Ensemble | À retenir |
|---|---|---|---|
| LAC | 1 − π̂_y | classes avec π̂ ≥ 1 − q̂ | Les plus petits ensembles, mais pas adaptatif. Ensembles vides possibles, taille moyenne < 1 si le modèle est très bon |
| APS | Masse cumulée des probas triées jusqu'à la vraie classe incluse | On ajoute des classes jusqu'à dépasser δ | Adaptatif. La version randomisée garde la classe frontière au hasard |
| RAPS | APS + une pénalité λ par classe au-delà du rang k_reg | Idem | Plus stable si le softmax est bruité |

## 5. Indices pour le TP (sans les solutions)
Le TP est en **TensorFlow**. Utilise le bon noyau et `%pip install puncc` dans le notebook.

Partie I (régression) :
- Données : X ~ U[0, 20], y = (1+ε)·X. Le bruit vaut ε·X, d'où l'hétéroscédasticité. scikit-learn veut X en 2D : `reshape(-1, 1)`.
- Split test 25 %, puis re-split du train en fit/calib à 50 %. **Entraîne sur X_fit, pas sur X_train.**
- `SplitConformal` : stocke score_fn, predset_fn, scores, quantile. `compute_quantile` utilise le niveau ⌈(1−α)(n+1)⌉/n avec `np.quantile(..., method="inverted_cdf")`, pour renvoyer un score réellement observé, comme dans la preuve.
- Évaluation : couverture = moyenne de (bas ≤ y ≤ haut). Attendu ≈ 0,9, avec un intervalle constant.
- CV+ : `KFold(shuffle=True, random_state=fixé)`, pour que le 2e `kf.split` redonne les mêmes folds. Chaque point est associé au modèle qui ne l'a pas vu.
- CQR : `GradientBoostingRegressor(loss="quantile", alpha=...)`. Ici, `alpha` de scikit-learn est le **niveau du quantile** (0,05 et 0,95), pas le α de la CP. Les commentaires du squelette de la cellule de calibration CQR sont inversés (upper/lower).

Partie II (MNIST, LAC) :
- CNN : 28 → 26 → 13 → 11 → 5, donc 5×5×32 = 800 entrées pour la Dense. Paramètres : 160 + 4 640 + 8 010.
- Piège : `compute_scores` appelle `score_fn(arg1, arg2)`. En régression c'est (y, ŷ) ; la solution LAC passe (probas, labels). Reste cohérent entre ta fonction et ton appel. Pour les scores, utilise les labels entiers, pas le one-hot.
