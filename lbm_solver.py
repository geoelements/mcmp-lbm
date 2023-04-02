#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
from matplotlib import style
from pylab import *
import os
import math
import vtk 
from vtk.util import numpy_support
from pyevtk.hl import gridToVTK
import taichi as ti
import json
import lbm_parameters as parameters
import utility as ut

# Default parameter settings
Q = 19 # number of discrete velocities
dim = 3 # dimensionality of simulation

# Adhesion and cohesion parameters
G = -1 # global cohesion parameter
G_1s = -0.4 # cohesion between fluid 1 and solid phase
G_12 = 4.0 # cohesion between fluid 1 and fluid 2
G_2s = -G_1s # cohesion between fluid 2 and solid phase
G_11 = 0 # intra-molecular interaction for fluid 1
G_22 = 0 # intra-molecular interaction for fluid 2

# Viscosity and initial density
niu_1 = 0.2 # viscosity of fluid 1
niu_2 = 0.2 # viscosity of fluid 2
rho_1 = 2 # initial density of fluid 1
rho_2 = 2 # initial density of fluid 2
rhos = 3 # density of solid phase

# Compute relaxation times and digonal collision matrix
tau_1f, tau_2f, S_dig_np_1, S_dig_np_2 = parameters.update_S(niu_1, niu_2)

# Injection control
inject_period = 100 # number of time steps between injections
density_increment = 0.0 # amount of density injected each time

# Pressure and pseudo density
IniPerturbRate = 1 # initial perturbation rate for pressure
carn_star = False  # if True, the fluid(s) has multiphase
T_Tc = 0.7 # temperature divided by critical temperature
rhol_spinodal = 0.2724 # maximum liquid density for Carnahan-Starling EOS
rhog_spinodal = 0.0484 # minimum vapor density for Carnahan-Starling EOS

# Output frequency
vtk_dstep = 50 # number of time steps between VTK output files
stat_dstep = 50 # number of time steps between statistics output files
vtk_path = "./vtks" # path to output VTK files
nb_steps = 1000 # total number of time steps in simulation

# Dimension
lx = ly = lz = 100 # size of simulation domain
R = 0 # place a sphere of fluid 1 with a radius of R in fluid 2

# Input files
solid_np = np.zeros((lx, ly, lz), dtype=np.int8) # solid phase initial configuration
nb_solid = 0 # number of solid nodes

# Operator type:
MRT = True # if True, use MRT operator; else, use BGK operator

# Multicomponent or singlecomponent
MC = False 

# Allocation of sparse memory
sparse = True

def read_json(json_file):
    """
    Reads the parameters from a JSON file and assigns them to global variables.

    Args:
    json_file (str): The path to the JSON file.

    Returns:
    None
    """
    global lx, ly, lz, G_12, niu_1, rho_1, G_1s, rho_2, niu_2, G_2s, rhos, R
    global tau_1f, tau_2f, S_dig_np_1, S_dig_np_2, solid_np, nb_solid
    global Q, dim, inject_period, density_increment, carn_star, T_Tc, G_11, G_22, MC, G
    global vtk_dstep, stat_dstep, nb_steps, vtk_path, sparse, MRT, rhol_spinodal, rhog_spinodal

    # Read the JSON file
    with open(json_file, 'r') as openfile:
        data = json.load(openfile)
        
    # Assign the values to global variables
    lx = data['size']['lx']
    ly = data['size']['ly']
    lz = data['size']['lz']

    # Fluid 1 parameters
    G_12 = data['fluid 1']['inter-molecule interaction']
    niu_1 = data['fluid 1']['viscosity']
    rho_1 = data['fluid 1']['density']
    G_1s = data['fluid 1']['solid cohesion']
    G_11 = data['fluid 1']['intra-molecule interaction'] 

    # Fluid 2 parameters
    rho_2 = data['fluid 2']['density']
    niu_2 = data['fluid 2']['viscosity']
    G_2s = data['fluid 2']['solid cohesion']
    G_22 = data['fluid 2']['intra-molecule interaction']

    # Update the D2Q9 relaxation parameters for both fluids
    tau_1f, tau_2f, S_dig_np_1, S_dig_np_2 = parameters.update_S(niu_1, niu_2)
    
    # Solid parameters
    rhos = data['solid']['density']
    solid_np = np.zeros((lx, ly, lz), dtype=np.int8)
    scale_factor = data['solid']['scale factor']

    if data['solid']['grain location']:
         # Read the grain location from a file
        solid_np = ut.read_positions(data['solid']['grain location'],lx,ly,lz,dim,scale_factor)
    elif data['solid']['voxel location']:
        # Read the voxel location from a ASCII file
        solid_np = ut.read_voxels(data['solid']['voxel location'],scale_factor)
    elif data['solid']['read tiff']:
        # Read the voxel location from a TIFF file
        solid_np = ut.read_tiff(data['solid']['read tiff'],lx,ly,lz,scale_factor)

    nb_solid = np.count_nonzero(solid_np)

        
    # Simulation parameters
    density_increment = data['setting']['inject amount']
    MC = data['setting']['multicomponent'] 
    vtk_dstep = data['setting']['vtk period']
    stat_dstep = data['setting']['stat period']
    nb_steps = data['setting']['number of steps']
    vtk_path = data['setting']['output folder']

    # Create the output folder if it doesn't exist
    if not os.path.exists(vtk_path):
        os.makedirs(vtk_path)
    sparse = data['setting']['sparse memory']
    MRT = data['setting']['MRT']
    R = data['setting']['fluid sphere radius']

    carn_star = data['EOS']['Carnahan Starling']
    T_Tc = data['EOS']['T_Tc']
    rhol_spinodal = data['EOS']['maximum liquid density']
    rhog_spinodal = data['EOS']['minimum vapor density']

@ti.func
def Press(rho_value) -> ti.f32:
    """
    Computes the pressure of a given density using either the Carnahan-Starling equation of state or the 
    ideal gas equation of state, depending on the value of the static variable 'carn_star'.
    
    Parameters:
        rho_value (float): The density value for which to compute the pressure
        
    Returns:
        float: The computed pressure value
    """
    
    if ti.static(carn_star):
        # Compute pressure using the Carnahan-Starling equation of state
        a = 1.0
        b = 4.0
        R = 1.0
        Tc = 0.0943
        
        # Compute temperature and viscosity
        T = T_Tc * Tc
        eta = b * rho_value / 4.0
        eta2 = eta * eta
        eta3 = eta2 * eta
        rho2 = rho_value * rho_value
        one_minus_eta = 1.0 - eta
        one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta
        
        # Compute and return pressure
        return (
            rho_value * R * T * (1 + eta + eta2 - eta3) / one_minus_eta3 - a * rho2
        )
    else:
        # Compute pressure using the ideal gas equation of state
        cs2 = 1.0 / 3.0
        psi = 1.0 - ti.exp(-rho_value)
        psi2 = psi * psi
        
        # Compute and return pressure
        return cs2 * rho_value + cs2 * G_1s / 2 * psi2


# This function calculates the Intra-component psi value, which is used for
# calculations within the same fluid component.
@ti.func
def Intra_psi(rho_value) -> ti.f32:
    # Set the sound speed squared
    cs2 = 1.0 / 3.0
    # Calculate the pressure based on the density
    p = Press(rho_value)
    # Calculate the Intra-component psi value
    Intra_psi = ti.sqrt(-2.0 * (p - cs2 * rho_value) / cs2)
    return Intra_psi


# These functions calculate the Inter-component psi values, which are used for
# calculations between different fluid components.
@ti.func
def Inter_psi(rho_2_value) -> ti.f32:
    # Calculate Inter-component psi
    return 1.0 - ti.exp(-rho_2_value)

@ti.data_oriented
class D3Q19_MC:
    def __init__(self, sparse_mem=True):
        """
        Initializes a 3D D3Q19 lattice Boltzmann simulation with two immiscible fluids.
        :param sparse_mem: a bool value indicating whether to use a sparse memory layout
        """
        # Initialize fields
        self.nb_solid_nodes = ti.field(ti.i32, shape=())
        self.step = ti.field(ti.i32, shape=())
        self.step[None] = 0
        self.IS_SOLID = ti.field(ti.i32)
        self.pressure = ti.field(ti.f32)

        self.collide_f_1 = ti.Vector.field(Q, ti.f32)
        self.stream_f_1 = ti.Vector.field(Q, ti.f32)
        self.rho_1 = ti.field(ti.f32)
        self.v_1 = ti.Vector.field(dim, ti.f32)
        self.force_1 = ti.Vector.field(dim, ti.f32)
        self.psi_1 = ti.field(ti.f32)
       
        self.collide_f_2 = ti.Vector.field(Q, ti.f32)
        self.stream_f_2 = ti.Vector.field(Q, ti.f32)
        self.rho_2 = ti.field(ti.f32)
        self.v_2 = ti.Vector.field(dim, ti.f32)
        self.force_2 = ti.Vector.field(dim, ti.f32)
        self.psi_2 = ti.field(ti.f32)

         # Set up memory layout
        if sparse_mem == False:
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.force_1,
                self.psi_1,
                self.rho_1,
                self.v_1,
                self.force_2,
                self.psi_2,
                self.rho_2,
                self.v_2,
                self.IS_SOLID,
                self.collide_f_1,
                self.collide_f_2,
                self.stream_f_1,
                self.stream_f_2,
                self.pressure,
            )
        else:
            n_mem_partition = 2  # Generate blocks of 2X2x2
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.IS_SOLID,
                self.psi_1,
                self.rho_1,
                self.psi_2,
                self.rho_2,
            )
            cell = ti.root.pointer(
                ti.ijk,
                (
                    lx // n_mem_partition + 1,
                    ly // n_mem_partition + 1,
                    lz // n_mem_partition + 1,
                ),
            )
            cell.dense(
                ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
            ).place(
                self.force_1,
                self.v_1,
                self.force_2,
                self.v_2,
                self.collide_f_1,
                self.collide_f_2,
                self.stream_f_1,
                self.stream_f_2,
                self.pressure,
            )

        # Set up constants for relaxation time
        self.w = ti.field(ti.f32, shape=(Q))
        self.w.from_numpy(parameters.w)
        self.M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
        self.inv_M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
        self.M_mat[None] = ti.Matrix(parameters.M_np)
        self.inv_M_mat[None] = ti.Matrix(parameters.inv_M_np)
        self.S_dig_vec_1 = ti.Vector.field(Q, ti.f32, shape=())
        self.S_dig_vec_1.from_numpy(S_dig_np_1)
        self.S_dig_vec_2 = ti.Vector.field(Q, ti.f32, shape=())
        self.S_dig_vec_2.from_numpy(S_dig_np_2)
        self.e_xyz = ti.Vector.field(dim, dtype=ti.i32, shape=(Q))
        self.e_xyz.from_numpy(np.array(parameters.e_xyz_list))
        self.ef_xyz = ti.Vector.field(dim, dtype=ti.f32, shape=(Q))
        self.ef_xyz.from_numpy(np.array(parameters.e_xyz_list))

        # REVERSED_E stores the index of the opposite component to every component in e_xyz_np
        # For example, for [1,0,0], the opposite component is [-1,0,0] which has the index of 2 in e_xyz
        self.REVERSED_E = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
        ]

        ti.static(self.REVERSED_E)
        self.IS_SOLID.from_numpy(solid_np)
        self.Gl = ti.field(ti.f32, shape=(3))
        self.Go = ti.field(ti.f32, shape=(3))

        self.Gl[0] = G_12
        self.Gl[1] = G_1s
        self.Gl[2] = G_11 # if G_11 is not zero, the first fluid is multiphase

        self.Go[0] = G_12
        self.Go[1] = G_2s
        self.Go[2] = G_22 # if G_22 is not zero, the second fluid is multiphase

        ti.static(self.inv_M_mat)
        ti.static(self.IS_SOLID)
        ti.static(self.S_dig_vec_2)
        ti.static(self.S_dig_vec_1)
        ti.static(self.M_mat)
        ti.static(self.e_xyz)
        ti.static(self.ef_xyz)
        ti.static(self.w)

        ti.static(self.Gl)
        ti.static(self.Go)

    @ti.kernel
    def update_psi(self):
        """
        Update the pseudo potential at current fluid node.
        """
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                # Update psi_1 and psi_2 fields based on the local density of fluid cells.
                self.psi_1[I] = Inter_psi(self.rho_1[I])
                self.psi_2[I] = Inter_psi(self.rho_2[I])
                # Update pressure field using Kruger's equation.
                self.pressure[I] = (
                    parameters.cs2 * (self.rho_1[I] + self.rho_2[I])
                    + parameters.cs2 * G_12 / 2 * self.psi_1[I] * self.psi_2[I]
                )


    @ti.kernel
    def update_force(self):
        """
        Updates the force vectors for each fluid node based on its neighboring nodes.
        """
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                # Initialize force vectors for fluid 1 and fluid 2 to zero.
                force_1 = ti.Vector([0.0, 0.0, 0.0])
                force_2 = ti.Vector([0.0, 0.0, 0.0])

                for i in ti.static(range(dim)):
                    for s in ti.static(range(1, Q)):
                        # Calculate the inter-component forces between fluid 1 and fluid 2 or between fluid 1 and the solid.
                        neighbor_pos = self.periodic_index(I + self.e_xyz[s])
                        force_1[i] += (
                            -self.psi_2[neighbor_pos]
                            * self.psi_1[I]
                            * self.w[s]
                            * self.Gl[self.IS_SOLID[neighbor_pos]]
                            * self.ef_xyz[s][i]
                        )
                        # Calculate the intra-component force if fluid 1 is multiphase.
                        force_1[i] += (
                            -Intra_psi(self.rho_1[neighbor_pos])
                            * Intra_psi(self.rho_1[I])
                            * self.w[s]
                            * self.Gl[2]
                            * self.ef_xyz[s][i]
                            * (1 - self.IS_SOLID[neighbor_pos])
                        )

                        # Calculate the inter-component force between fluid 2 and fluid 1 or between fluid 2 and the solid
                        force_2[i] += (
                            -self.psi_1[neighbor_pos]
                            * self.psi_2[I]
                            * self.w[s]
                            * self.Go[self.IS_SOLID[neighbor_pos]]
                            * self.ef_xyz[s][i]
                        )
                        # Calculate the intra-component force if fluid 2 is multiphase.
                        force_2[i] += (
                            -Intra_psi(self.rho_2[neighbor_pos])
                            * Intra_psi(self.rho_2[I])
                            * self.w[s]
                            * self.Go[2]
                            * self.ef_xyz[s][i]
                            * (1 - self.IS_SOLID[neighbor_pos])
                        )

                # Assign the calculated force vectors to the fluid nodes.
                self.force_1[I] = force_1
                self.force_2[I] = force_2

    @ti.kernel
    def update_velocity(self):
        """
        Updates the velocity for each fluid node based on its pdf and force.
        """
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:

                for i in ti.static(range(dim)):
                    temp_vel_1 = 0.0
                    temp_vel_2 = 0.0
                    for s in ti.static(range(Q)):
                        temp_vel_1 += (
                            self.collide_f_1[I][s] * self.ef_xyz[s][i] / tau_1f
                        )
                        temp_vel_2 += (
                            self.collide_f_2[I][s] * self.ef_xyz[s][i] / tau_2f
                        )

                    self.v_1[I][i] = (temp_vel_1 + self.force_1[I][i] * tau_1f) / self.rho_1[I]
                    self.v_2[I][i] = (temp_vel_2 + self.force_2[I][i] * tau_2f) / self.rho_2[I]
                        
    @ti.func
    def meq_vec(self, rho_local, u):
        out = ti.Vector(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # The D3Q19 Gram-Schmidt equilibrium moments
        out[0] = 1
        out[3] = u[0]
        out[5] = u[1]
        out[7] = u[2]
        out[1] = u.dot(u)
        out[9] = 2 * u.x * u.x - u.y * u.y - u.z * u.z
        out[11] = u.y * u.y - u.z * u.z
        out[13] = u.x * u.y
        out[14] = u.y * u.z
        out[15] = u.x * u.z
        return out * rho_local

    @ti.func
    def place_fluid_sphere(self, x, y, z, R):
        """
        Place a sphere of fluid 1 in fluid 2 (to conduct droplet test).
        """
        xmin = x - R
        ymin = y - R
        zmin = z - R
        xmax = x + R
        ymax = y + R
        zmax = z + R
        nb_nodes = 0
        for px in range(xmin, xmax + 1):
            for py in range(ymin, ymax + 1):
                for pz in range(zmin, zmax + 1):
                    dx = px - x
                    dy = py - y
                    dz = pz - z
                    dist2 = dx * dx + dy * dy + dz * dz
                    R2 = R * R
                    if dist2 < R2:
                        near_px = ti.floor(px + 0.5, ti.i32)
                        near_py = ti.floor(py + 0.5, ti.i32)
                        near_pz = ti.floor(pz + 0.5, ti.i32)
                        if ti.floor(px + 0.5) <= 0:
                            near_px = ti.floor(px + 0.5, ti.i32) + lx
                        if ti.floor(py + 0.5) <= 0:
                            near_py = ti.floor(py + 0.5, ti.i32) + ly
                        if ti.floor(pz + 0.5) <= 0:
                            near_pz = ti.floor(pz + 0.5, ti.i32) + lz

                        if near_px >= lx:
                            near_px -= lx
                        if near_py >= ly:
                            near_py -= ly
                        if near_pz >= lz:
                            near_pz -= lz

                        if self.IS_SOLID[near_px, near_py, near_pz] == 0 and near_pz<lz/2:
                            self.rho_1[near_px, near_py, near_pz] = rho_1
                            self.rho_2[near_px, near_py, near_pz] = rho_2/20

                            nb_nodes += 1
        return nb_nodes

    @ti.kernel
    def init_field(self):
        for x, y, z in self.IS_SOLID:
            self.rho_1[x, y, z] = rhos
            self.rho_2[x, y, z] = rhos
            self.psi_1[x, y, z] = Inter_psi(rhos)
            self.psi_2[x, y, z] = Inter_psi(rhos)

            if self.IS_SOLID[x, y, z] == 0:
                self.rho_1[x, y, z] = rho_1/20
                self.rho_2[x, y, z] = rho_2
        
        # Measure the contact angle by conducting a droplet test
        if R>0:
            for x in range(lx):
                for y in range(ly):
                    self.IS_SOLID[x, y, lz - 1] = 1
                    self.IS_SOLID[x, y, 0] = 1
            self.place_fluid_sphere(int(lx/2), int(ly/2), R, R+1)
             

        for x, y, z in self.IS_SOLID:
            if self.IS_SOLID[x, y, z] == 0:
                self.psi_1[x, y, z] = Inter_psi(self.rho_1[x, y, z])
                self.psi_2[x, y, z] = Inter_psi(self.rho_2[x, y, z])

                for q in ti.static(range(Q)):
                    self.collide_f_1[x, y, z][q] = parameters.t[q] * self.rho_1[x, y, z]
                    self.stream_f_1[x, y, z][q] = parameters.t[q] * self.rho_1[x, y, z]

                    self.collide_f_2[x, y, z][q] = parameters.t[q] * self.rho_2[x, y, z]
                    self.stream_f_2[x, y, z][q] = parameters.t[q] * self.rho_2[x, y, z]

    # check if sparse storage works!
    @ti.kernel
    def activity_checking(self) -> int:
        nb_activated_nodes = 0
        for x, y, z in self.collide_f_1:
            nb_activated_nodes += 1
        return nb_activated_nodes

    @ti.kernel
    def collision(self):
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                if ti.static(MRT):
                    #MRT operator
                    m_1 = self.M_mat[None] @ self.collide_f_1[I]
                    m_eq_1 = self.meq_vec(self.rho_1[I], self.v_1[I])
                    m_1 -= self.S_dig_vec_1[None] * (m_1 - m_eq_1)

                    m_2 = self.M_mat[None] @ self.collide_f_2[I]
                    m_eq_2 = self.meq_vec(self.rho_2[I], self.v_2[I])
                    m_2 -= self.S_dig_vec_2[None] * (m_2 - m_eq_2)

                    self.collide_f_1[I] = ti.Vector(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    self.collide_f_1[I] += self.inv_M_mat[None] @ m_1

                    self.collide_f_2[I] = ti.Vector(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    self.collide_f_2[I] += self.inv_M_mat[None] @ m_2

                else:
                    # using BGK operator
                    u_squ_1 = self.v_1[I].dot(self.v_1[I])
                    for s in ti.static(range(Q)):
                        eu = self.ef_xyz[s].dot(self.v_1[I])
                        self.collide_f_1[I][s] += (
                            parameters.t[s]
                            * self.rho_1[I]
                            * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_squ_1)
                            - self.collide_f_1[I][s]
                        ) / tau_1f
                    
                    u_squ_2 = self.v_2[I].dot(self.v_2[I])
                    for s in ti.static(range(Q)):
                        eu = self.ef_xyz[s].dot(self.v_2[I])
                        self.collide_f_2[I][s] += (
                            parameters.t[s]
                            * self.rho_2[I]
                            * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_squ_2)
                            - self.collide_f_2[I][s]
                        ) / tau_2f

    @ti.kernel
    def post_collsion(self):
        """
        Update density distribution after after collision.
        """
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                self.collide_f_1[I] = self.stream_f_1[I]
                self.rho_1[I] = self.collide_f_1[I].sum()

                self.collide_f_2[I] = self.stream_f_2[I]
                self.rho_2[I] = self.collide_f_2[I].sum()

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = lx - 1
        if i[0] > lx - 1:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = ly - 1
        if i[1] > ly - 1:
            iout[1] = 0
        if i[2] < 0:
            iout[2] = lz - 1
        if i[2] > lz - 1:
            iout[2] = 0

        return iout

    @ti.kernel
    def streaming(self):
        """
        Propogate the density distribution to neighboring fluid nodes.
        """
        for i in ti.grouped(self.collide_f_1):
            if self.IS_SOLID[i] == 0 and i.x < lx and i.y < ly and i.z < lz:
                for s in ti.static(range(Q)):
                    ip = self.periodic_index(i + self.e_xyz[s])
                    if self.IS_SOLID[ip] == 0:
                        self.stream_f_1[ip][s] = self.collide_f_1[i][s]
                        self.stream_f_2[ip][s] = self.collide_f_2[i][s]
                    else:
                        self.stream_f_1[i][self.REVERSED_E[s]] = self.collide_f_1[i][s]
                        self.stream_f_2[i][self.REVERSED_E[s]] = self.collide_f_2[i][s]

    def export_VTK(self, n):
        """
        Export the fluid field to VTK.
        """
        x = np.linspace(0, lx, lx)
        y = np.linspace(0, ly, ly)
        z = np.linspace(0, lz, lz)
        grid_x = np.linspace(0, lx, lx)
        grid_y = np.linspace(0, ly, ly)
        grid_z = np.linspace(0, lz, lz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        gridToVTK(
            vtk_path +'vkt_'+str(n),
            grid_x,
            grid_y,
            grid_z,
            pointData={
                "Solid": np.ascontiguousarray(
                    self.IS_SOLID.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "rho_1": np.ascontiguousarray(self.rho_1.to_numpy()[0:lx, 0:ly, 0:lz]),
                "rho_2": np.ascontiguousarray(self.rho_2.to_numpy()[0:lx, 0:ly, 0:lz]),
                "pressure": np.ascontiguousarray(
                    self.pressure.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "velocity_1": (
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "velocity_2": (
                    np.ascontiguousarray(self.v_2.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.v_2.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.v_2.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "psi_1": np.ascontiguousarray(self.psi_1.to_numpy()[0:lx, 0:ly, 0:lz]),
                "psi_2": np.ascontiguousarray(self.psi_2.to_numpy()[0:lx, 0:ly, 0:lz]),
                "force_1": (
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "force_2": (
                    np.ascontiguousarray(self.force_2.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.force_2.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.force_2.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
            },
        )

    def run(self, max_step=20000):
                
        while self.step[None] < max_step:
            if (self.step[None]) % vtk_dstep == 0:
                self.export_VTK(self.step[None] // vtk_dstep)
                print(
                    "Export No.{} vtk at step {}".format(
                        self.step[None] // vtk_dstep, self.step[None]
                    )
                )
            
            self.update_psi()
            self.update_force()
            self.update_velocity()
            self.collision()
            self.streaming()
            self.post_collsion()

            self.step[None] += 1

@ti.data_oriented
class D3Q19_SC:
    def __init__(self, sparse_mem=True):

        self.nb_solid_nodes = ti.field(ti.i32, shape=())
        self.step = ti.field(ti.i32, shape=())
        self.step[None] = 0
        self.IS_SOLID = ti.field(ti.i32)

        self.collide_f_1 = ti.Vector.field(Q, ti.f32)
        self.stream_f_1 = ti.Vector.field(Q, ti.f32)
        self.rho_1 = ti.field(ti.f32)
        self.v_1 = ti.Vector.field(dim, ti.f32)
        self.force_1 = ti.Vector.field(dim, ti.f32)
        self.psi_1 = ti.field(ti.f32)
        self.pressure = ti.field(ti.f32)

        if sparse_mem == False:
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.force_1,
                self.psi_1,
                self.rho_1,
                self.v_1,
                self.IS_SOLID,
                self.collide_f_1,
                self.stream_f_1,
                self.pressure,
            )

        else:
            n_mem_partition = 2  # Generate blocks of 3X3x3
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.IS_SOLID,
                self.psi_1,
                self.rho_1,
            )
            cell = ti.root.pointer(
                ti.ijk,
                (
                    lx // n_mem_partition + 1,
                    ly // n_mem_partition + 1,
                    lz // n_mem_partition + 1,
                ),
            )
            cell.dense(
                ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
            ).place(
                self.force_1,
                self.v_1,
                self.collide_f_1,
                self.stream_f_1,
                self.pressure,
            )

        # to compare if disassembled for-loop is faster or not
        self.w = ti.field(ti.f32, shape=(Q))
        self.w.from_numpy(parameters.w)

        self.M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
        self.inv_M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())

        self.M_mat[None] = ti.Matrix(parameters.M_np)
        self.inv_M_mat[None] = ti.Matrix(parameters.inv_M_np)

        self.S_dig_vec_1 = ti.Vector.field(Q, ti.f32, shape=())
        self.S_dig_vec_1.from_numpy(S_dig_np_1)

        self.e_xyz = ti.Vector.field(dim, dtype=ti.i32, shape=(Q))
        self.e_xyz.from_numpy(np.array(parameters.e_xyz_list))

        self.ef_xyz = ti.Vector.field(dim, dtype=ti.f32, shape=(Q))
        self.ef_xyz.from_numpy(np.array(parameters.e_xyz_list))

        # REVERSED_E stores the index of the opposite component to every component in e_xyz_np
        # For example, for [1,0,0], the opposite component is [-1,0,0] which has the index of 2 in e_xyz
        self.REVERSED_E = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
        ]

        ti.static(self.REVERSED_E)
        self.IS_SOLID.from_numpy(solid_np)

        ti.static(self.IS_SOLID)
        ti.static(self.S_dig_vec_1)
        ti.static(self.e_xyz)
        ti.static(self.ef_xyz)
        ti.static(self.w)
    
    @ti.func
    def place_fluid_sphere(self, x, y, z, R,rho_value):
        xmin = x - R
        ymin = y - R
        zmin = z - R
        xmax = x + R
        ymax = y + R
        zmax = z + R
        nb_nodes = 0
        for px in range(xmin, xmax + 1):
            for py in range(ymin, ymax + 1):
                for pz in range(zmin, zmax + 1):
                    dx = px - x
                    dy = py - y
                    dz = pz - z
                    dist2 = dx * dx + dy * dy + dz * dz
                    R2 = R * R
                    if dist2 < R2:
                        near_px = ti.floor(px + 0.5, ti.i32)
                        near_py = ti.floor(py + 0.5, ti.i32)
                        near_pz = ti.floor(pz + 0.5, ti.i32)
                        if ti.floor(px + 0.5) <= 0:
                            near_px = ti.floor(px + 0.5, ti.i32) + lx
                        if ti.floor(py + 0.5) <= 0:
                            near_py = ti.floor(py + 0.5, ti.i32) + ly
                        if ti.floor(pz + 0.5) <= 0:
                            near_pz = ti.floor(pz + 0.5, ti.i32) + lz
                        if near_px >= lx:
                            near_px -= lx
                        if near_py >= ly:
                            near_py -= ly
                        if near_pz >= lz:
                            near_pz -= lz
                        if self.IS_SOLID[near_px, near_py, near_pz] == 0:
                            self.rho_1[near_px, near_py, near_pz] = rho_value
                            nb_nodes += 1
        return nb_nodes
    

    @ti.func
    def force_vec(self, local_pos) -> parameters.f32_vec3d:
        force_vec = ti.Vector([0.0, 0.0, 0.0])
        local_psi = Intra_psi(self.rho_1[local_pos])
        for i in ti.static(range(3)):
            for s in ti.static(range(1, Q)):
                neighbor_pos = self.periodic_index(local_pos + self.e_xyz[s])
                neighbor_psi = Intra_psi(self.rho_1[neighbor_pos])
                force_vec[i] += self.w[s] * neighbor_psi * self.ef_xyz[s][i]
        force_vec *= local_psi
        return force_vec

    @ti.func
    def velocity_vec(self, local_pos) -> parameters.f32_vec3d:
        velocity_vec = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for s in ti.static(range(Q)):
                velocity_vec[i] += self.collide_f_1[local_pos][s] * self.ef_xyz[s][i]

            velocity_vec[i] += self.force_1[local_pos][i] * tau_1f
            velocity_vec[i] /= self.rho_1[local_pos]
        return velocity_vec

    @ti.func
    def meq_vec(self, rho_local, u):
        out = ti.Vector(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # The D3Q19 Gram-Schmidt equilibrium moments
        out[0] = 1
        out[3] = u[0]
        out[5] = u[1]
        out[7] = u[2]
        out[1] = u.dot(u)
        out[9] = 2 * u.x * u.x - u.y * u.y - u.z * u.z
        out[11] = u.y * u.y - u.z * u.z
        out[13] = u.x * u.y
        out[14] = u.y * u.z
        out[15] = u.x * u.z
        return out * rho_local

    def export_VTK(self, n):
        x = np.linspace(0, lx, lx)
        y = np.linspace(0, ly, ly)
        z = np.linspace(0, lz, lz)
        grid_x = np.linspace(0, lx, lx)
        grid_y = np.linspace(0, ly, ly)
        grid_z = np.linspace(0, lz, lz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        gridToVTK(
            vtk_path +"SC_" + str(n),
            grid_x,
            grid_y,
            grid_z,
            pointData={
                "Solid": np.ascontiguousarray(
                    self.IS_SOLID.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "rho_1": np.ascontiguousarray(self.rho_1.to_numpy()[0:lx, 0:ly, 0:lz]),
                "pressure": np.ascontiguousarray(
                    self.pressure.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "velocity_1": (
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.v_1.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "force_1": (
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.force_1.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
            },
        )

    @ti.kernel
    def init_field(self):
        for x, y, z in self.IS_SOLID:
            if self.IS_SOLID[x, y, z] == 0:
                self.rho_1[x,y,z] = rho_2
                # rho0 * (1.0 + IniPerturbRate * (ti.random(ti.f32) - 0.5))
            else:
                self.rho_1[x, y, z] = rhos

        self.place_fluid_sphere(50, 50, 10, 11, 0.3)

        for x, y, z in self.IS_SOLID:
            if self.IS_SOLID[x, y, z] == 0:
                for q in ti.static(range(Q)):
                    self.collide_f_1[x, y, z][q] = parameters.t[q] * self.rho_1[x, y, z]
                    self.stream_f_1[x, y, z][q] = parameters.t[q] * self.rho_1[x, y, z]

    @ti.kernel
    def collision(self):
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:

                self.force_1[I] = self.force_vec(I)
                self.v_1[I] = self.velocity_vec(I)

                if ti.static(MRT):
                    """MRT operator"""
                    # Matrix dot product
                    m = self.M_mat[None] @ self.collide_f_1[I]
                    m_eq = self.meq_vec(self.rho_1[I], self.v_1[I])
                    m -= self.S_dig_vec[None] * (m - m_eq)

                    self.collide_f_1[I] = ti.Vector(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    self.collide_f_1[I] += self.inv_M_mat[None] @ m

                else:
                    """BGK operator"""
                    u_squ = self.v_1[I].dot(self.v_1[I])
                    for s in ti.static(range(Q)):
                        eu = self.ef_xyz[s].dot(self.v_1[I])
                        self.collide_f_1[I][s] += (
                            parameters.t[s]
                            * self.rho_1[I]
                            * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_squ)
                            - self.collide_f_1[I][s]
                        ) / tau_1f

    @ti.kernel
    def post_collsion(self):
        """Calculate force and velocity"""
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                self.collide_f_1[I] = self.stream_f_1[I]
                self.rho_1[I] = self.collide_f_1[I].sum()

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = lx - 1
        if i[0] > lx - 1:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = ly - 1
        if i[1] > ly - 1:
            iout[1] = 0
        if i[2] < 0:
            iout[2] = lz - 1
        if i[2] > lz - 1:
            iout[2] = 0

        return iout

    @ti.kernel
    def streaming(self):
        for I in ti.grouped(self.collide_f_1):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                for s in ti.static(range(19)):
                    neighbor_pos = self.periodic_index(I + self.e_xyz[s])
                    if self.IS_SOLID[neighbor_pos] == 0:
                        self.stream_f_1[neighbor_pos][s] = self.collide_f_1[I][s]
                    else:
                        self.stream_f_1[I][self.REVERSED_E[s]] = self.collide_f_1[I][
                            s
                        ]

    def run(self, max_step=1000):
        
        print("The number of solid nodes is ", nb_solid)
        print("Porosity = {:.3f}".format(1 - nb_solid / (lx * ly * lz)))

        self.export_VTK(self.step[None] // vtk_dstep)

        while self.step[None] < max_step:

            if (self.step[None]) % vtk_dstep == 0 and self.step[None]>0:
                self.export_VTK(self.step[None] // vtk_dstep)
                print(
                    "Export No.{} vtk at step {}".format(
                        self.step[None] // vtk_dstep, self.step[None]
                    )
                )

            self.collision()
            self.streaming()
            self.post_collsion()
            self.step[None] += 1

            