# LOG — synth-graph-rs (Ilias)

---

## Semaine 1

Début du projet. J'ai d'abord lu quelques papiers sur le SBM pour comprendre le modèle
avant d'écrire quoi que ce soit. La structure est simple : on partitionne N nœuds en K
blocs, puis pour chaque paire on tire une arête avec probabilité p_in si même bloc,
p_out sinon.

J'ai mis en place le projet Rust avec maturin/PyO3 et fait un premier `generate_sbm`
qui fonctionne. Le problème : pour N=10000 c'est vraiment lent parce que je parcours
toutes les N*(N-1)/2 paires. Il faut trouver mieux.

---

## Semaine 2

Pour accélérer, j'ai implémenté l'échantillonnage géométrique : au lieu de tester
chaque paire, on tire directement le *gap* jusqu'à la prochaine arête à partir d'une
variable géométrique. Ça passe de O(N²) à O(E). Le seul truc compliqué était de
décoder l'indice linéaire dans le triangle supérieur — j'ai fait une recherche binaire
sur la formule cumulative.

Ajout du DC-SBM en même temps : chaque nœud reçoit un θ suivant une loi de Pareto,
et les probabilités d'arêtes sont pondérées par θ_u * θ_v. On normalise θ par bloc
pour garder le degré moyen cohérent.

---

## Semaine 3

Implémentation du cSBM. En plus des arêtes, chaque nœud reçoit un vecteur de features.
La logique : chaque classe a un centroïde, et les features = centroïde + bruit. J'ai
supporté trois distributions de bruit (gaussienne, uniforme, laplacienne).

Wilfried a besoin que les graphes soient connexes pour sa visualisation, donc j'ai
ajouté `ensure_connected` qui relie les composantes déconnectées par des arêtes de
pont. J'ai aussi ajouté `p_triangle` pour fermer des triangles après coup — ça rend
les graphes plus réalistes (coefficient de clustering plus élevé).

Aussi ajouté `feat_noise_ratio` : une fraction des nœuds reçoit les features d'une
mauvaise classe, pour simuler du bruit de label.

---

## Semaine 4

Travail sur la sérialisation JSON pour respecter le contrat avec Wilfried. Le format
est : metadata (n_nodes, n_edges, homophily), nodes (id, community, features?),
edges (source, target). J'ai utilisé serde pour ça.

Côté Python j'ai aussi fait `json_to_pyg` qui reprend un JSON et sort les tableaux
numpy au format PyG. Et `generate_from_config` qui accepte directement le JSON que
Wilfried envoie depuis son TUI — comme ça il n'a pas à connaître les détails de
l'API interne.

---

## Semaine 5

Revue du code avant rendu. Plusieurs corrections :

- Validation des entrées : j'avais oublié de vérifier que `avg_degree` et `mu`
  ne sont pas NaN/infini. Ajout de gardes `is_finite()`.
- Le tri des class_weights utilisait `partial_cmp` ce qui peut planter sur NaN.
  Remplacé par `total_cmp`.
- Renommage des variables : plein de noms d'un seul caractère (`n`, `c`, `h`, `d`,
  `f`...) remplacés par des noms descriptifs partout, y compris dans les signatures
  Python exposées.
- Ajout des doc comments sur toutes les fonctions et types (exigence du cours).
- 36 tests unitaires pour couvrir les cas limites principaux.
