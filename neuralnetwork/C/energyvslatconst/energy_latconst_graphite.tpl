################################################################################
# Compute the energy of a graphite with a lattice parameter of a=2.466 and
# c=3.348 Angstrom. Periodic boundary conditions are applied in all directions.
# The crystal is relaxed in the z direction.
################################################################################

# Initialize interatomic potential (KIM model) and units
atom_style	atomic
kim init	{{ potential }} metal
# boundary conditions
boundary	p p p
# create a honeycomb lattice
variable	a0 equal {{ a }}
variable	c0 equal 3.348
variable	coa equal ${c0}/${a0}
lattice		custom ${a0} a1 1.0 0.0 0.0 &
			  a2 0.5 $(sqrt(3.0)/2.0) 0.0 &
			  a3 0.0 0.0 ${coa} &
			  basis 0.0 0.0 0.0 &
			  basis $(1.0/3.0) $(1.0/3.0) 0.0 &
			  spacing 1.0 $(sqrt(3.0)/2.0) ${coa}
# create simulation box and atoms
region		reg prism 0 1 0 1 0 1 0.5 0 0 units lattice
create_box	1 reg
create_atoms	1 box
mass		1 12
# specify atom type to chemical species mapping for the KIM model
kim interactions C
kim param	set active_member_id 1 {{ set_id }}
# minimization over the z direction
fix		1 all box/relax z 0.0
thermo_style	custom step temp etotal press lx ly lz
run		0
min_style	cg
min_modify	line backtrack
minimize	1e-25 1e-25 50000 100000
# extract
variable 	natoms equal "count(all)" 
variable	Eng equal pe/${natoms}
variable	c equal lz