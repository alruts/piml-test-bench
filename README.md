*To-do list*

**Open questions**
- Is there *any* way to do differentiable single frequency acoustic FEM/BEM or do I have to implement this? Extend jWave?
- Are neural pdes actually viable ways of solving inverse problems (from kidger: PINNs are a terrible idea, 
- Consult with Allan to see if he thinks neural PDE are feasible to solve inverse problems in acoustics.

**General**
-[ ] Look at Python code for optimal sensor placement by `@manoharDataDrivenSparseSensor2018.`
    -[ ] Write a module to interface with the test simulation test benches, to determine sensor placements.
-[ ] Write an email to Patrick Kidger to get his opinion on nde for inverse problems
-[ ] Do a pull request on jWave to see if I can implement general boundary conditions (state-space model)
and velocity sources.

**Loudspeaker test**
-[ ] Write a differentiable simulation for a moving piston loudspeaker in `diffrax`.
    -[ ] Make dataset with this and test with PINN based method.
    -[ ] Make optimal sensor placement tests.
    -[ ] Make Neural PDE approach (parameterize the source as a network).
-[ ] Do measurements in anechoic chamber (good to have researched sensor placements via simulations in advance).
-[ ] Compare results from measured and simulated datasets.

**Impedance test**
-[ ] Reuse some of the things from thesis, implement a complex-valued alternative.
    -[ ] Ask Matteo for advice on activation functions / initialization schemes.
-[ ] 
-[ ] 
-[ ] 

# Test bench

This is a project to implement a test bench for inverse problems in acoustics.

Problems considered are
1. Boundary condition estimation
2. Source characterization

These two problems are tightly coupled, as they both deal with estimating the
state of a surface from a set of pressure measurements in its vicinity.

To characterize a surface acoustically, we generally need two state variables:
- Pressure $p$ (scalar field)
- Velocity $\boldsymbol{v}$ (vector-field)

Which together form the impedance field

$$\boldsymbol{Z} = p / \boldsymbol{v}$$

Generally, can view problems 1 and 2 as impedance estimation tasks where
boundary conditions span the space of passive impedance fields and sources the
space of active impedance fields. 

<!-- passive/active classification --> 
The criterion ...

$$|\frac{\zeta + 1}{\zeta - 1}| >= 1$$

where ... are considered active and ... are considered passive.

## Optimal sensor placement

The test bench is also meant to be used to research optimal sensor placements
for such problems,

## Test Cases

Each case has a physical test bench set up at DTU, with a corresponding
simulation setup implemented in ...

### Active case: A loudspeaker in free-field


### Passive case: A squared absorber in free-field


## Solution architectures

1. Linear method (such as DMD?)
2. PINN for inverse estimation
    - Vanilla PINN
    - CV PINN
3. Neural differential equation methods
    - Idea to model the known dynamics via PDE, with unknown source term as network.
    - Idea to incorporate stochastic PDE for non-deterministic sources (tire-road).

## Meeting notes

### 16/09/2025
- Andreas could not join due to dentist appointment.
- Sturla settling in at DTU, computer set up.
- Leaning towards simulations in the frequency domain.
    - Saves computational load.
    - This is what we are interested in the end.
- Discussed neural PDE approach, want to research this direction.
    - Pushing the state-of-the-art if it works.
- PINN requires a lot of guidance via custom loss/architectures.
- Next time there will be some slides.

