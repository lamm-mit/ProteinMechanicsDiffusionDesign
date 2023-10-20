import os
import sys

import pandas as pd
import torch

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import linecache
import re

from Bio.PDB import PDBParser, PDBIO
import math

from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.vectors import calc_angle, calc_dihedral
import Bio.PDB.vectors
#
from Bio.PDB.DSSP import DSSP # add try a self-made one
# from Bio.PDB.DSSP_SelfMade import DSSP_SelfMade # add try a self-made one

resdict = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
# using those from force field file
#
resdict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",

}
#
# SMD setup
SMD_Vel = 0.0001 # A/timestep

# step_data * SMD_Vel = pulling_dist

def collect_geo_of_backbone(chain):
    prev = "0"
    rad = 180.0 / math.pi
    # result
    resu = {"AA":[],\
            "Bond_CA_N":[],"Bond_CA_C":[],"Bond_N_C1":[],\
            "Angl_CA1_C1_N":[],"Angl_C1_N_CA":[],"Angl_N_CA_C":[],\
            "Dihe_PHI":[],"Dihe_PSI":[],"Dihe_OME":[]}
    #
    for res in chain:
        if res.get_resname() in resdict.keys():

            # seq += resdict[res.get_resname()]
            resu["AA"].append(resdict[res.get_resname()])
            # ToDo, check whether this res has N, CA, C
#             if not (res.has_key("N") and res.has_key("NA") and res.has_key("C")):
#                 print("Key backbone atom is missing")

            if prev == "0":
                # 1st AA:
                N_prev = res["N"]
                CA_prev = res["CA"]
                C_prev = res["C"]
                # update the key
                prev = "1"
            else:
                n1 = N_prev.get_vector()
                ca1 = CA_prev.get_vector()
                c1 = C_prev.get_vector()
                
                # print(res)
                C_curr = res["C"]
                N_curr = res["N"]
                CA_curr = res["CA"]

                # get the coordinates
                c = C_curr.get_vector()
                n = N_curr.get_vector()
                ca = CA_curr.get_vector()

                # get the measurement
                ca1_c1_n_ThisAngle = calc_angle(ca1, c1, n)*rad
                c1_n_ca_ThisAngle = calc_angle(c1, n, ca)*rad
                n_ca_c_ThisAngle = calc_angle(n, ca, c)*rad

                ca_n_ThisBond = CA_curr - N_curr
                ca_c_ThisBond = CA_curr - C_curr
                n_c1_ThisBond = N_curr - C_prev

                ThisPsi = calc_dihedral(n1, ca1, c1, n)    # degree
                ThisOmega = calc_dihedral(ca1, c1, n, ca)  # degree
                ThisPhi = calc_dihedral(c1, n, ca, c)      # degree

                # store the results
                # n1-ca1-c1--n-ca-c--n2-ca2-c2
                resu["Bond_CA_N"].append(ca_n_ThisBond)
                resu["Bond_CA_C"].append(ca_c_ThisBond)
                resu["Bond_N_C1"].append(n_c1_ThisBond) # peptide bond
                #
                resu["Angl_CA1_C1_N"].append(ca1_c1_n_ThisAngle)
                resu["Angl_C1_N_CA"].append(c1_n_ca_ThisAngle)
                resu["Angl_N_CA_C"].append(n_ca_c_ThisAngle)
                #
                resu["Dihe_PHI"].append(ThisPhi)
                resu["Dihe_PSI"].append(ThisPsi)
                resu["Dihe_OME"].append(ThisOmega)

                # update the AA info
                N_prev = res["N"]
                CA_prev = res["CA"]
                C_prev = res["C"]

    # summerize the result
    return resu
#
def collect_multi_chain_AA_info(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("sample", pdb_file)
    resu_full = {"Chain":[],"AA":{}}
    for chain in structure.get_chains():
        this_chain_id = chain.get_id()
        # print('Working on Chain ', this_chain_id)
        # working on one chain; Assume there is only one chain
        resu_full["Chain"].append(this_chain_id)
        resu_test = collect_geo_of_backbone(chain)
        resu_full["AA"][this_chain_id]=resu_test["AA"]
        # can add more
    
    return resu_full



# read one record

# plot one record ONLY in the non-empty cases
#
def get_one_force_record(ii, resu_file_name_list):
    # ii = pick_file_list[i]
    pdb_id = resu_file_name_list['PDB_ID'][ii]
    data_one_file = resu_file_name_list['Path'][ii]+'/1_working_dir/collect_results/smd_resu.dat'
    data = np.genfromtxt(data_one_file)
    # print(data.shape)
    # kernel = np.ones(kernel_size) / kernel_size
    
    # focus on disp-force curve
    # print('# of data point: ', data.shape[0])
    disp_data = data[:,1]
    force_data = data[:,7]
    
    # + add the pulling point info
    # pulling point disp
    step_data = data[:,0]
    setdata_one_file = resu_file_name_list['Path'][ii]+'/1_working_dir/box_dimension_after_eq.dat'
    line_4 = linecache.getline(setdata_one_file, 4)
    SMD_Vel = float(line_4.split()[2])
    pull_data = SMD_Vel*step_data
    
    # force_data_convolved_10 = np.convolve(force_data, kernel, mode='same')
    return disp_data, force_data, pdb_id, pull_data

# collect AA from the record
def get_one_AA_record(ii, resu_file_name_list):
    # ii = pick_file_list[i]
    # TestProt_chain_0_after_psf.pdb
    pdb_file = resu_file_name_list['Path'][ii]+'/1_working_dir/TestProt_chain_0_after_psf.pdb'
    
    resu_full = collect_multi_chain_AA_info(pdb_file)
    # Here, we assume there is only one chain in the file, which is the case for tensile test
    # AA_seq = resu_full["AA"][resu_full["Chain"][0]]
    AA_seq = ''.join(resu_full["AA"][resu_full["Chain"][0]])
    
    return AA_seq

# smooth functions
def conv_one_record(force_data, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    force_data_convolved = np.convolve(force_data, kernel, mode='same')
    
    return force_data_convolved

from math import factorial

from scipy.ndimage.filters import uniform_filter1d
#
# function to smooth the data    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    
    try:
        # window_size = np.abs(np.int(window_size))
        window_size = np.abs(int(window_size))
        # order = np.abs(np.int(order))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

#
def read_gap_values_from_dat(file):
#     line_2 = linecache.getline('r"'+file+'"', 2)
#     line_3 = linecache.getline('r"'+file+'"', 3)
    line_2 = linecache.getline(file, 2)
    line_3 = linecache.getline(file, 3)
    # get the values
    ini_gap = float(line_2.split()[2])
    fin_gap = float(line_3.split()[2])
    return ini_gap, fin_gap


def read_one_array_from_df(one_record):
    return np.array(list(map(float, one_record.split(" "))))
#
def read_string_find_max(reco):
    x = read_one_array_from_df(reco)
    return np.amax(x)

def read_string_find_max(reco):
    x = read_one_array_from_df(reco)
    return np.amax(x)
#
def cal_seq_end_gap(x):
    inc_gap_arr = x['posi_data']-x['posi_data'][0]
    ini_gap = x['ini_gap']
    gap_arr = ini_gap+inc_gap_arr
    
    return gap_arr
#
def cal_pull_end_gap(x):
    inc_gap_arr = x['pull_data'] # -x['pull_data'][0]
    ini_gap = x['ini_gap']
    gap_arr = ini_gap+inc_gap_arr
    
    return gap_arr

#
# pick the force at the unfolding of every residues

def simplify_NormPull_FORCEnF_rec(n_fold,this_seq_len,this_n_PullGap_arr,this_Force_arr):

    target_pull_gap_list = [1./(this_seq_len*n_fold)*(jj+0) for jj in range(this_seq_len*n_fold)]
    target_pull_gap_list.append(1.)

    # retrive the force values
    target_force = []
    for jj in range(len(target_pull_gap_list)):
    # for jj in range(10):
        this_t_n_PullGap = target_pull_gap_list[jj]

        if this_t_n_PullGap<this_n_PullGap_arr[0]:
            this_t_F = 0.
        else:
            # find the neareast one
            disp_arr = np.abs(this_n_PullGap_arr - this_t_n_PullGap)
            pick_id = np.argmin(disp_arr)
            this_t_F = this_Force_arr[pick_id]
        #
        target_force.append(this_t_F)
    #
    target_pull_gap_arr = np.array(target_pull_gap_list)
    target_force_arr = np.array(target_force)
    
    # for delivery 
    resu = {}
    resu['sample_NormPullGap'] = target_pull_gap_arr
    resu['smaple_FORCE'] = target_force_arr
    return resu

# read input conditions
def read_input_model_A(file_path):
    with open(file_path, 'r') as f:
        txt = f.read()
    nums = re.findall(r'\[([^][]+)\]', txt)
    arr = np.loadtxt(nums)
    # print(arr)
    # print(arr[0])

    return arr

def read_input_model_B(file_path):
    with open(file_path, 'r') as f:
        txt = f.read()
    nums = re.findall(r'\[([^][]+)\]', txt)
    # arr = np.loadtxt(nums)
    arr = np.loadtxt( [nums[0].replace('\n','')] )
    # print(arr)
    # print(arr[0])

    return arr

def read_one_input_arr_from_txt(file_path):
    with open(file_path, 'r') as f:
        txt = f.read()
    nums = re.findall(r'\[([^][]+)\]', txt)
    # arr = np.loadtxt(nums)
    arr = np.loadtxt( [nums[0].replace('\n','')] )
    # print(arr)
    # print(arr[0])

    return arr

# this only for this version, in folder3 it is updated
# # for folder3
# def recover_input_for_model_B(file_path, seq_len):
#     raw_arr = read_one_input_arr_from_txt(file_path)
#     arr = raw_arr[1:1+seq_len+1]
#     return arr
# for folder2
def recover_input_for_model_B_ver2(file_path, seq_len):
    raw_arr = read_one_input_arr_from_txt(file_path)
    arr = raw_arr[0:0+seq_len+1]
    return arr

# for folder3
def recover_input_for_model_B_ver3(file_path, seq_len):
    raw_arr = read_one_input_arr_from_txt(file_path)
    arr = np.zeros(seq_len+1)
    arr[1:1+seq_len] = raw_arr[0:0+seq_len]
    return arr