# PDE-Surrogate-RandomForest-XGBoost
"Modèles substituts (Surrogate Models) basés sur Random Forest et XGBoost pour la résolution numérique d'équations aux dérivées partielles (EDP)."

# Modèles Substituts (Surrogate Models) pour la Résolution d'EDP Non Linéaires

Ce projet présente une approche de Machine Learning (Random Forest et XGBoost) pour prédire en temps réel la solution faible d'une Équation aux Dérivées Partielles (EDP) non linéaire, en s'appuyant sur un cadre d'analyse variationnelle rigoureux et une validation numérique par la Méthode des Éléments Finis (MEF) sous MATLAB.

-----------------------------------------------------------------------------------------------------------------------------------------

## 1. Formulation Analytique du Problème

On considère le problème aux limites non linéaire suivant dans l'intervalle $\Omega = (0,1)$ :

$$-\Delta u + u^3 = f \quad \text{dans } (0,1)$$

avec les conditions aux limites de Dirichlet homogènes : $u(0) = u(1) = 0$. 

Dans la suite, on fixe le terme source $f(x) = \sin(\pi x)$, vérifiant $\|\sin(\pi x)\|_{L^2} = \frac{1}{\sqrt{2}}$.

### 1.1 Choix du Cadre Fonctionnel
Pour définir l'espace de travail des solutions faibles, nous analysons séparément l'opérateur différentiel et le terme non linéaire :
* **L'opérateur de Laplace ($-\Delta$)** : Associé aux conditions de Dirichlet homogènes, il impose que les fonctions ainsi que leurs dérivées premières soient de carré intégrable. Cela nous oriente vers l'espace de Sobolev classique $H_0^1(0,1)$.
* **Le terme non linéaire ($u^3$)** : Pour que ce terme ait un sens dans la formulation variationnelle lorsqu'il est multiplié par une fonction test $v \in H_0^1(0,1)$, il faut que $u^3 \in L^2(0,1)$, ce qui implique $u \in L^6(0,1)$.

**Théorème d'injection de Sobolev (Dimension $N=1$) :** En dimension 1, l'espace $H^1(0,1)$ s'injecte de façon continue (et compacte) dans l'espace des fonctions continues $C^0([0,1])$, et par conséquent dans tous les espaces $L^p(0,1)$ pour tout $p \geq 1$.

Puisque $H_0^1(0,1) \hookrightarrow L^6(0,1)$ de manière continue, l'espace idoine pour ce problème est tout simplement :

$$V = H_0^1(0,1)$$

muni de sa norme standard équivalente (grâce à l'inégalité de Poincaré) :
$$\|u\|_V = \left( \int_0^1 |u'(x)|^2 \, dx \right)^{1/2}$$

### 1.2 Formulation Variationnelle
Soit $v \in H_0^1(0,1)$ une fonction test. En multipliant l'EDP par $v$ et en intégrant par parties, nous obtenons le problème variationnel suivant : Trouver $u \in H_0^1(0,1)$ tel que :

$$\mathcal{A}(u,v) = \mathcal{L}(v), \quad \forall v \in H_0^1(0,1)$$

Où l'opérateur non linéaire $\mathcal{A}(u,v)$ et la forme linéaire $\mathcal{L}(v)$ sont définis par :
$$\mathcal{A}(u,v) = \int_0^1 u' v' \, dx + \int_0^1 u^3 v \, dx$$
$$\mathcal{L}(v) = \int_0^1 \sin(\pi x) v \, dx$$

---

## 2. Existence et Unicité de la Solution Faible

Le théorème de Lax-Milgram ne s'appliquant pas directement à cause de la non-linéarité du terme $u^3$, nous utilisons la méthode variationnelle directe (minimisation de la fonctionnelle d'énergie) :

$$J(u) = \frac{1}{2}\int_0^1 |u'|^2 \, dx + \frac{1}{4}\int_0^1 u^4 \, dx - \int_0^1 \sin(\pi x) u \, dx$$

### A) Coercivité de la fonctionnelle $J$
Le terme non linéaire étant positif ($\int_0^1 u^4 \, dx \geq 0$), en utilisant l'inégalité de Cauchy-Schwarz et de Poincaré ($\|u\|_{L^2} \leq C_p \|u'\|_{L^2}$), nous parvenons à :

$$J(u) \geq \frac{1}{2} \|u\|_V^2 - \|\sin(\pi x)\|_{L^2} \|u\|_{L^2} \geq \frac{1}{2} \|u\|_V^2 - \frac{C_p}{\sqrt{2}} \|u\|_V$$

Quand $\|u\|_V \to +\infty$, le terme quadratique domine, d'où $\lim_{\|u\|_V \to +\infty} J(u) = +\infty$ (Coercivité).

### B) Semi-continuité inférieure faible (s.c.i.f.)
Soit $(u_n)$ une suite convergeant faiblement vers $u$ dans $H_0^1(0,1)$. 
* Par la s.c.i.f de la norme pour la topologie faible : $\liminf_{n \to \infty} \|u_n\|_V^2 \geq \|u\|_V^2$.
* Grâce à l'injection compacte $H_0^1(0,1) \hookrightarrow L^4(0,1)$, la convergence faible devient forte dans $L^4$, assurant la convergence exacte du terme non linéaire. 

D'où $\liminf_{n \to +\infty} J(u_n) \geq J(u)$.

### C) Conclusion
La fonctionnelle $J$ étant coercive, strictement convexe et s.c.i.f sur un espace de Banach réflexif, elle admet un unique minimum global $u \in H_0^1(0,1)$, qui est l'unique solution faible du problème (Théorème de Stampacchia / Minty-Browder).

---

## 3. Propriété Qualitative : Principe du Maximum (Positivité)

### Théorème
La solution faible $u \in H_0^1(0,1)$ du problème vérifie : $u(x) \geq 0$ presque partout dans $(0,1)$.

### Démonstration (Méthode de troncature de Stampacchia)
1. **Choix de la fonction test** : On introduit la partie négative $u^-(x) = \max(0, -u(x))$. Par le théorème de Stampacchia, $u^- \in H_0^1(0,1)$ et son gradient vaut $-\nabla u$ si $u < 0$, et $0$ sinon. On l'utilise comme fonction test ($v = u^-$).
2. **Injection dans la formulation faible** :
   $$\int_0^1 u' (u^-)' \, dx + \int_0^1 u^3 u^- \, dx = \int_0^1 \sin(\pi x) u^- \, dx$$
3. **Analyse des termes** :
   * Terme de diffusion : $\int_0^1 u' (u^-)' \, dx = -\int_0^1 |(u^-)'(x)|^2 \, dx$
   * Terme non linéaire : $\int_0^1 u^3 u^- \, dx = -\int_0^1 |u^-(x)|^4 \, dx$
4. **Estimation finale** : En multipliant par $-1$, l'égalité devient :
   $$\underbrace{\int_0^1 |(u^-)'(x)|^2 \, dx + \int_0^1 |u^-(x)|^4 \, dx}_{\geq 0} = \underbrace{-\int_0^1 \sin(\pi x) u^-(x) \, dx}_{\leq 0}$$
Pour respecter les signes, il est mathématiquement nécessaire que les deux membres soient rigoureusement nuls. On en déduit que $\int_0^1 |u^-(x)|^4 \, dx = 0$, d'vou $u^-(x) = 0$ presque partout. 

**Conclusion** : $u(x) = u^+(x) \geq 0$ presque partout sur $(0,1)$. $\blacksquare$

---

## 💻 4. Simulations et Validations Numériques (MATLAB)

Afin de valider empiriquement le comportement théorique de notre modèle, plusieurs scripts de calcul numérique ont été développés sous MATLAB :

### 4.1 Résolution par Éléments Finis $P_1$ & Algorithme de Newton-Raphson
La minimisation de la fonctionnelle d'énergie $J(u)$ équivaut à résoudre le système algébrique non linéaire issu de la discrétisation spatiale :
$$\mathbf{K} \cdot \mathbf{U} + \mathbf{N\_vec}(\mathbf{U}) = \mathbf{F}$$
Un solveur itératif basé sur la méthode de **Newton-Raphson** a été implémenté pour traiter le terme cubique $u^3$. Le modèle converge de manière quadratique en seulement 4 à 5 itérations (Tolérance = $10^{-8}$).

### 4.2 Analyse de Convergence Énergétique et d'Erreurs
* **Étude Énergétique** : Calcul et suivi de l'énergie minimale $J(u_h)$ sur des maillages de plus en plus fins ($N = 10$ à $320$). On observe une stabilisation parfaite de l'énergie vers la borne inférieure absolue.
* **Analyse d'Erreur et EOC (Experimental Order of Convergence)** : En utilisant une solution de référence hautement résolue ($N_{ref} = 640$), nous avons calculé les erreurs dans les normes classiques. Les résultats confirment strictement les estimations d'erreur a priori de la théorie des Éléments Finis :
  * Ordre de convergence dans la norme $L^2$ : $\mathcal{O}(h^2)$ (pente expérimentale $\approx 2.00$)
  * Ordre de convergence dans la semi-norme $H^1$ : $\mathcal{O}(h)$ (pente expérimentale $\approx 1.00$)

### 4.3 Générateur de Données (Data Factory pour l'IA)
Un script d'automatisation a été conçu pour générer le dataset d'entraînement. En simulant $50$ fonctions sources $f(x)$ aléatoires (combinaisons de sinus et de polynômes) et en résolvant à chaque fois le problème par la MEF, nous avons extrait un fichier structuré `pde_data_multi.csv` contenant **5100 lignes** de features géométriques et physiques.

---

## 🛠️ 5. Implémentation du Machine Learning & Déploiement

Ce dataset propre et stable, issu du solveur physique MATLAB, a été injecté dans un environnement de Data Science pour entraîner des modèles prédictifs rapides :
1. **Random Forest Regressor**
2. **XGBoost Regressor**

### 🚀 Application Interactive (Streamlit Cloud)
Les modèles entraînés sont déployés au sein d'une interface utilisateur web interactive via **Streamlit**. Elle permet de fournir n'importe quel terme source $f(x)$ et de prédire/visualiser **instantanément** (en une fraction de seconde, sans aucune itération lourde) le profil complet de la solution faible $u(x)$.




