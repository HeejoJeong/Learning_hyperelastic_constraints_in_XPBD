# Learning Neural **Hyper-Elastic** Constraints in XPBD
*From a few frames, learning a reusable constraint directly inside XPBD.*
<p align="center">
  <img src="GT.gif"  width="45%"/>
  <img src="gen.gif" width="45%"/>
  <br>
  <em>Learning from target Bunny (40 frames) → Generating Armadillo scene!</em>
</p>

---

## 1. XPBD

- XPBD is a unified real-time solver that handles diverse materials such as cloth, soft bodies, and fluids.  
- It is highly efficient, stable, GPU-friendly, and widely used in interactive graphics and simulation.  
- Physical behaviour is captured **by imposing constraint functions** $C(\mathbf x)=0$.  
- For details, see the original [XPBD paper](https://matthias-research.github.io/pages/publications/XPBD.pdf) by Macklin et al., 2016.

---

## 2. Our approach
- **Mesh-based neural constraint**  
  * An MLP imposes the constraint on each tetrahedron in the mesh by outputting a scalar residual, replacing hand-crafted formulas.
    
- **Built-in physics priors** (inspired by [NCLaw](https://sites.google.com/view/nclaw) (Ma et al., 2023)):  
  1. **Rotation / translation invariance**  
     * Feed the network only *deformation-invariant* scalars:
       $[tr\Sigma(F)-3,\lVert F\rVert_F^{2}-3,\det(F)-1]$, where $F$ is the deformation gradient.
     * Any rigid motion leaves these inputs unchanged, so the learned constraint is automatically invariant.  
  2. **Zero energy in the undeformed state**  
     * Use a **bias-free** MLP → when all three inputs are zero (undeformed), the output residual is forced to zero.
## 3. Result

- **Data-efficient training(Bunny)**  
  * Three short **Bunny** (~1.7 k tets) trajectories, 40 frames each, with parameters (Δt, E, ν):
    * 🟢 green - (0.006 s, 1500, 0.475)
    * 🔵 red - (0.010 s, 2000, 0.35)
    * 🔴 blud - (0.013 s, 1333, 0.33)
  * Each constraint is trained on its own trajectory (3 models total).  
  * Training uses **15 XPBD solver iterations** per time step.  
  * All trajectories were generated with a stable Neo-Hookean material.
  * Because some time steps demand more solver iterations than others, we **adaptively adjust the iteration count** during training so the network sees both stiff (many-iteration) and easy (few-iteration) cases.

- **Generalization(Armadilo)**  
  * Plug the learned constraints into XPBD and run on an **unseen Armadillo mesh** (≈ 10 k tets) colliding with a floor.  
  * Use a **coarser time step**, $\Delta t = 0.02 s$, with **20 XPBD solver iterations.**  


## 4.setup

We used an **RTX 3090** GPU, with **Python 3.9**, **PyTorch 2.5.0**, and **CUDA Toolkit 11.8**.
We also used [Taichi](https://www.taichi-lang.org/) and [Mesh Taichi](https://github.com/taichi-dev/meshtaichi).