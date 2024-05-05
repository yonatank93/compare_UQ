################################################################################
# Compute the energy of a diamond carbon with a lattice parameter of a=3.56
# Angstrom. Periodic boundary conditions are applied in all directions.
################################################################################

# Initialize interatomic potential (KIM model) and units
atom_style	atomic
kim init	{{ potential }} metal
# boundary conditions
boundary	p p p
# create a honeycomb lattice
lattice		diamond {{ a }}
# create simulation box and atoms
region		reg block 0 1 0 1 0 1 units lattice
create_box	1 reg
create_atoms	1 box
mass		1 12
# specify atom type to chemical species mapping for the KIM model
kim interactions C
kim param	set active_member_id 1 {{ set_id }}
# compute energy
run		0
variable	natoms equal 2
variable	E_mean equal pe/${natoms}