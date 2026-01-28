# Feuille de route - Correction & Évolution du Projet

Basé sur l'analyse du code effectuée le 23 Janvier 2026.

## 1. Corrections Critiques (Pourquoi l'accuracy stagne)

### A. Double Normalisation des Données (Prioritaire)
**Problème :** Les données d'entrée sont divisées par 255 deux fois (une fois dans le parsing CSV, une fois dans `get_train`). Les valeurs résultantes (~0.000015) sont trop faibles pour l'apprentissage.
- [x] **Action :** Dans `packages/core/src/internal/dataset.rs`, supprimer la division dans `read_csv` OU la normalisation dans `get_train`. Ne garder qu'une seule mise à l'échelle (0-1).

### B. Calcul du Loss et Dimensions
**Problème :** Le calcul de l'erreur (`calculate_loss`) utilise des dimensions incorrectes, mélangeant le batch et les classes.
- [x] **Action :** Dans `packages/core/src/neural.rs` -> `calculate_loss` :
    - Transposer les prédictions `a` (`a.t()?`) avant de les passer à `cross_entropy`.
    - Changer `y.t()?.argmax(0)?` en `y.t()?.argmax(1)?` (ou s'assurer que les dimensions correspondent à `(Batch,)` pour les cibles).

### C. Backpropagation de la dernière couche
**Problème :** Le calcul du gradient initial `dz = a - y` suppose une activation Softmax combinée à une CrossEntropy, mais la dernière couche actuelle n'a pas d'activation (`Activation::None`).
- [ ] **Action :** Ajouter `Activation::Softmax` à la dernière couche lors de la construction du réseau dans `packages/core/src/main.rs`.

---

## 2. Améliorations Architecturales & Clean Code

### A. Refactoring "Rust Idiomatic"
**Problème :** Utilisation de la structure `Computer` comme une classe statique style Java.
- [ ] **Action :** Supprimer `struct Computer`. Déplacer la logique directement dans les méthodes de `impl Layer` ou des fonctions libres privées.
- [ ] **Action :** Remplacer les assertions de forme (`assert_shape`) par des vérifications d'erreur standard au début des fonctions.

### B. Gestion des Hyperparamètres
**Problème :** Valeurs en dur (Hardcoded).
- [ ] **Action :** Rendre le nombre de classes dynamique (actuellement `max = 100` en dur pour le One-Hot encoding).
- [ ] **Action :** Implémenter un `DataLoader` pour gérer des mini-batchs au lieu de charger tout le dataset en mémoire (Full Batch).

### C. Évolution vers Candle Autodiff
**Problème :** Réimplémentation manuelle de la backpropagation limitante pour les architectures complexes (Transformers, etc.).
- [ ] **Action :** Pour les futurs modèles, utiliser le moteur d'autodifférenciation de Candle (`Var`, `backward()`) au lieu de calculer manuellement les gradients (`dws`, `dbs`). Garder l'implémentation manuelle actuelle comme référence pédagogique.

### D. Séparation des Responsabilités
- [ ] **Action :** Extraire la logique d'entraînement (`train` loop) hors de la struct `Network` vers une struct `Trainer` dédiée.
