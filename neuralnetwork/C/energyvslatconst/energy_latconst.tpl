################################################################################
# Compute the energy of a monolayer graphene with a lattice parameter of a=2.466
# Angstrom. Periodic boundary conditions are applied in the two in-plane
# directions, while the out-of-plane direction is free.
#
# The energy is computed using the DUNN potential with 100 ensemble members
# (each with different dropout matrices). The mean and standard deviation of
# the energy are reported.
################################################################################

# Initialize interatomic potential (KIM model) and units
atom_style	atomic
kim init	{{ potential }} metal
# boundary conditions
boundary	p p f
# create a honeycomb lattice
variable	a equal {{ a }}
lattice		custom $a a1 1.0 0.0 0.0 &
			  a2 0.5 $(sqrt(3.0)/2.0) 0.0 &
			  a3 0.0 0.0 1.0 &
			  basis 0.0 0.0 0.0 &
			  basis $(1.0/3.0) $(1.0/3.0) 0.0 &
			  spacing 1.0 $(sqrt(3.0)/2.0) 1.0
# create simulation box and atoms
region		reg prism 0 1 0 1 0 1 0.5 0 0 units lattice
create_box	1 reg
create_atoms	1 box
mass		1 12
# specify atom type to chemical species mapping for the KIM model
kim interactions C
# compute energy
run		0
variable	natoms equal 2
variable	E_mean equal pe/${natoms}