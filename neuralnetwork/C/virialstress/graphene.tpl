# Initialize interatomic potential (KIM model) and units
atom_style	atomic
kim             init {{ modelname }} metal
# boundary conditions
boundary	p p p
# create a honeycomb lattice
variable	a equal {{ latparam }}
variable	sqrt3 equal sqrt(3.0)
variable	b equal $a/${sqrt3}
variable	sc equal 3.0*$b  # Lattice scale
variable	hsc equal 34/${sc}  # Scaled van der Waals thickness
variable	asc equal $a/${sc}  # Scaled a
lattice		custom ${sc} &
			a1 1.0 0.0 0.0 &
			a2 0.0 ${asc} 0.0 &
			a3 0.0 0.0 ${hsc} &
			basis 0.0 0.0 0.0 &
			basis $(1.0/3.0) 0.0 0.0 &
			basis 0.5 0.5 0.0 &
			basis $(5.0/6.0) 0.5 0.0
# create simulation box and atoms
region		box block 0 4 0 6 -0.5 0.5 units lattice
create_box	1 box
create_atoms	1 box
mass		1 12

# specify atom type to chemical species mapping for the KIM model
kim		interactions C
{{ kim_param_str }}

# Equilibrate for 10_000 time steps
variable	temp equal 300.0
velocity        all create ${temp} 2023 mom yes rot yes
fix             mom all momentum 1 linear 1 1 1 angular
fix		mynve all nve
fix		mylgv all langevin 0.0 ${temp} 0.01 2023


thermo		100
thermo_style	custom step temp etotal press lx ly lz
run		10000
unfix           mylgv

# Start recording the data
compute         virial all pressure NULL virial
dump		mydump all custom 10 "{{ dumppath }}/virial.dump" id type x y z fx fy fz
dump_modify	mydump format float %.16f
# Using this dump format, I can use ase.io.read to read the file and get all the
# information I need, including the atomic positions and forces.

# MD
fix		mylgv all langevin ${temp} ${temp} 0.1 2023
reset_timestep	0
thermo		10
thermo_style	custom step temp etotal press lx ly lz c_virial[*]
run		10000

undump		mydump
