import Mathlib

variable {d : ℕ}

-- L-smoothness of F
def LSmooth (F : EuclideanSpace ℝ (Fin d) → ℝ) (L : ℝ) : Prop :=
  ∀ x y, ‖gradient F x - gradient F y‖ ≤ L * ‖x - y‖

noncomputable def projOntoX
    (X : Set (EuclideanSpace ℝ (Fin d)))
    (hne : X.Nonempty)
    (hcomplete : IsComplete X)
    (hconvex : Convex ℝ X)
    (y : EuclideanSpace ℝ (Fin d)) : EuclideanSpace ℝ (Fin d) :=
  Classical.choose (exists_norm_eq_iInf_of_complete_convex hne hcomplete hconvex y)

noncomputable def projGradMap
    (X : Set (EuclideanSpace ℝ (Fin d)))
    (hne : X.Nonempty)
    (hcomplete : IsComplete X)
    (hconvex : Convex ℝ X)
    (x g : EuclideanSpace ℝ (Fin d))
    (η : ℝ) : EuclideanSpace ℝ (Fin d) :=
  (1 / η) • (x - projOntoX X hne hcomplete hconvex (x - η • g))


