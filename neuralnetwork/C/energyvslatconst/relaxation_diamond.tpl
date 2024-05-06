# This script is used to find the equilibrium stucture of graphene.

# ---------- Initialize Simulation --------------------- 
clear 
kim init {{ potential }} metal
dimension 3 
boundary p p p 
atom_style atomic 
atom_modify map array

# ---------- Create Atoms --------------------- 
lattice		diamond {{ ainit }}

region box block 0 1 0 1 0 1 units lattice
create_box 1 box
create_atoms	1 box
replicate 1 1 1
mass 1 12.011

# ---------- Define Interatomic Potential ---------------------
kim interactions C
neighbor 2.0 bin 
neigh_modify delay 10 check yes 
# Set dropout active member
kim param set active_member_id 1 {{ active_id }}
 
# ---------- Define Settings --------------------- 
compute eng all pe/atom 
compute eatoms all reduce sum c_eng 

# ---------- Run Minimization --------------------- 
reset_timestep 0
fix 1 all box/relax iso 0.0 vmax 0.001
# dump Dump all cfg 10 dump/relaxation_*.cfg mass type xs ys zs fx fy fz
# dump_modify Dump   element C
thermo 10 
thermo_style custom step pe lx ly lz press pxx pyy pzz c_eatoms 
min_style cg
min_modify line backtrack
minimize 1e-25 1e-25 50000 100000
# undump Dump

variable natoms equal "count(all)" 
variable teng equal "c_eatoms"
variable length equal "ly"
variable ecoh equal "v_teng/v_natoms"

print "Total energy (eV) = ${teng};"
print "Number of atoms = ${natoms};"
print "Lattice constant (Angstoms) = ${length};"
print "Cohesive energy (eV) = ${ecoh};"

print "All done!" 
