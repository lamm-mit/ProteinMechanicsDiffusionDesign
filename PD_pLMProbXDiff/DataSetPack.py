from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from torch.utils.data import DataLoader,Dataset
import pandas as pd
import seaborn as sns

import torchvision
 
import matplotlib.pyplot as plt
import numpy as np
 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from functools import partial, wraps

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

from matplotlib.ticker import MaxNLocator

import torch

import esm

# ============================================================
# convert csv into df
# ============================================================
class RegressionDataset(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__ (self):
            return len(self.X_data)

# padding a list using a given value
def pad_a_np_arr(x0,add_x,n_len):
    n0 = len(x0)
    # print(n0)
    x1 = x0.copy()
    if n0<n_len:
        for ii in range(n0,n_len,1):
            # x1.append(add_x)
            x1= np.append(x1, [add_x])
    else:
        print('No padding is needed')
        
    return x1

# padding a list using a given value for esm
# # --
# # need one padding at the beginning: Not anymore
# def pad_a_np_arr_esm(x0,add_x,n_len):
    
#     # print(n0)
#     x1 = x0.copy()
#     # print('copy x0: ', x1)
#     # print('x0 len: ', len(x1))
#     # add one for <cls>
#     x1 = [add_x]+x1 # somehow, this one doesn't work
#     # print(x1)
#     # print('x1 len: ',len(x1) )
#     n0 = len(x1)
#     #
#     if n0<n_len:
#         for ii in range(n0,n_len,1):
#             # x1.append(add_x)
#             x1= np.append(x1, [add_x])
#     else:
#         print('No padding is needed')
        
#     return x1
# ++
def pad_a_np_arr_esm(x0,add_x,n_len):
    
    # print(n0)
    x1 = x0.copy()
    # print('copy x0: ', x1)
    # print('x0 len: ', len(x1))
    # # add one for <cls>
    # x1 = [add_x]+x1 # somehow, this one doesn't work
    # print(x1)
    # print('x1 len: ',len(x1) )
    n0 = len(x1)
    #
    if n0<n_len:
        for ii in range(n0,n_len,1):
            # x1.append(add_x)
            x1= np.append(x1, [add_x])
    else:
        print('No padding is needed')
        
    return x1

# ============================================================
# convert csv into df: new for smoothed data of SMD
# 1. screen the dataset
# ============================================================
def screen_dataset_MD(
    # # --
    # file_path,
    # ++
    csv_file=None,
    pk_file =None,
    PKeys=None,
    CKeys=None,
):
    # unload the parameters
    
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    min_AASeq_len = PKeys['min_AA_seq_len']
    max_AASeq_len = PKeys['max_AA_seq_len']
    max_used_Smo_F = PKeys['max_Force_cap']
    
    # working part
    if csv_file != None:
        # functions
        print('=============================================')
        print('1. read in the csv file...')
        print('=============================================')
        arr_key = PKeys['arr_key']
        
        df_raw = pd.read_csv(csv_file)
        print(df_raw.keys())

        # convert string array back to array
        for this_key in arr_key:
            # np.array(list(map(float, one_record.split(" "))))
            df_raw[this_key] = df_raw[this_key].apply(lambda x: np.array(list(map(float, x.split(" ")))))
        # patch up
        df_raw.rename(columns={"sample_FORCEpN_data":"sample_FORCE_data"}, inplace=True)
        print('Updated keys: \n', df_raw.keys())
    
    elif pk_file != None:
        df_raw = pd.read_pickle(pk_file)
        print(df_raw.keys())
    
    # ..............................................................................
    #
    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( df_raw )):
        if df_raw['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                df_raw['sample_NormPullGap_data'][ii], 
                # df_raw['sample_FORCEpN_data'][ii], 
                df_raw['sample_FORCE_data'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            ax0.scatter(
                df_raw['NPullGap_for_MaxSmoF'][ii], 
                df_raw['Max_Smo_Force'][ii], 
            )
        else:
            print(df_raw['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized distance btw pulling ends')
    plt.ylabel('Force (pF)')
    outname = store_path+'CSV_0_SMD_sim_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    
    print('=============================================')
    print('2. screen the entries...')
    print('=============================================')
    #
    df_isnull = pd.DataFrame(
        round(
            (df_raw.isnull().sum().sort_values(ascending=False)/df_raw.shape[0])*100,
            1
        )
    ).reset_index()
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    print('Check null...')
    print( df_isnull )
    
    print('Working on a dataframe with useful keywords')
    protein_df = pd.DataFrame().assign(
        pdb_id=df_raw['pdb_id'],
        AA=df_raw['AA'], 
        seq_len=df_raw['seq_len'],
        Max_Smo_Force=df_raw['Max_Smo_Force'],
        NPullGap_for_MaxSmoF=df_raw['NPullGap_for_MaxSmoF'],
        # # --
        # sample_FORCEpN_data=df_raw['sample_FORCEpN_data'],
        # ++
        sample_FORCE_data=df_raw['sample_FORCE_data'],
        sample_NormPullGap_data=df_raw['sample_NormPullGap_data'],
        ini_gap=df_raw['ini_gap'],
        Fmax_pN_BSDB=df_raw['Fmax_pN_BSDB'],
        # ++ add some for energy integration
        pull_data=df_raw['pull_data'],
        forc_data=df_raw['forc_data'],
        Int_Smo_ForcPull=df_raw['Int_Ene'],
    )
    # ++ add new keys on energy
    
    # screen using AA length
    print('a. screen using sequence length...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['seq_len']>max_AASeq_len-2].index, 
        inplace = True
    )
    protein_df.drop(
        protein_df[protein_df['seq_len'] <min_AASeq_len].index, 
        inplace = True
    )
    protein_df=protein_df.reset_index(drop=True)
    print('used sequences #: ', len(protein_df))
    
    print('b. screen using force values...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['Max_Smo_Force']>max_used_Smo_F].index, 
        inplace = True
    )
    # protein_df.drop(
    #     protein_df[protein_df['seq_len'] <min_AA_len].index, 
    #     inplace = True
    # )
    protein_df=protein_df.reset_index(drop=True)
    print('afterwards, sequences #: ', len(protein_df))
    
    
    
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    sns.distplot(
        protein_df['seq_len'],
        bins=50,kde=False, 
        rug=False,norm_hist=False)
    sns.distplot(
        df_raw['seq_len'],
        bins=50,kde=False, 
        rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('AA length')
    outname = store_path+'CSV_1_AALen_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    #
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    sns.distplot(protein_df['Max_Smo_Force'],kde=True, rug=False,norm_hist=False)
    sns.distplot(df_raw['Max_Smo_Force'],kde=True, rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('Max. Force (pN) from MD')
    outname = store_path+'CSV_2_MaxSmoF_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    
    # re-plot the simplified results of SMD
    print('Check selected in SMD records...')

    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( protein_df )):
        if protein_df['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                protein_df['sample_NormPullGap_data'][ii], 
                # protein_df['sample_FORCEpN_data'][ii], 
                protein_df['sample_FORCE_data'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            ax0.scatter(
                protein_df['NPullGap_for_MaxSmoF'][ii], 
                protein_df['Max_Smo_Force'][ii], 
            )
        else:
            print(protein_df['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized distance btw pulling ends')
    plt.ylabel('Force (pF)')
    outname = store_path+'CSV_3_Screened_SMD_sim_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    
    # ++
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    sns.distplot(protein_df['Int_Smo_ForcPull'],kde=True, rug=False,norm_hist=False)
    sns.distplot(df_raw['Int_Ene'],kde=True, rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('Integrated energy (pN*Angstrom) from smoothed MD hist.')
    outname = store_path+'CSV_4_MaxSmoF_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print('Done')
    
    
    return df_raw, protein_df

# ............................................................
def screen_dataset_MD_old(
    file_path,
    PKeys=None,
    CKeys=None,
):
    # unload the parameters
    arr_key = PKeys['arr_key']
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    min_AASeq_len = PKeys['min_AA_seq_len']
    max_AASeq_len = PKeys['max_AA_seq_len']
    max_used_Smo_F = PKeys['max_Force_cap']
    
    # working part
    # functions
    print('=============================================')
    print('1. read in the csv file...')
    print('=============================================')
    df_raw = pd.read_csv(file_path)
    print(df_raw.keys())
    
    # convert string array back to array
    for this_key in arr_key:
        # np.array(list(map(float, one_record.split(" "))))
        df_raw[this_key] = df_raw[this_key].apply(lambda x: np.array(list(map(float, x.split(" ")))))
    
    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( df_raw )):
        if df_raw['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                df_raw['sample_NormPullGap_data'][ii], 
                df_raw['sample_FORCEpN_data'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            ax0.scatter(
                df_raw['NPullGap_for_MaxSmoF'][ii], 
                df_raw['Max_Smo_Force'][ii], 
            )
        else:
            print(df_raw['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized distance btw pulling ends')
    plt.ylabel('Force (pF)')
    outname = store_path+'CSV_0_SMD_sim_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    
    print('=============================================')
    print('2. screen the entries...')
    print('=============================================')
    #
    df_isnull = pd.DataFrame(
        round(
            (df_raw.isnull().sum().sort_values(ascending=False)/df_raw.shape[0])*100,
            1
        )
    ).reset_index()
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    print('Check null...')
    print( df_isnull )
    
    print('Working on a dataframe with useful keywords')
    protein_df = pd.DataFrame().assign(
        pdb_id=df_raw['pdb_id'],
        AA=df_raw['AA'], 
        seq_len=df_raw['seq_len'],
        Max_Smo_Force=df_raw['Max_Smo_Force'],
        NPullGap_for_MaxSmoF=df_raw['NPullGap_for_MaxSmoF'],
        sample_FORCEpN_data=df_raw['sample_FORCEpN_data'],
        sample_NormPullGap_data=df_raw['sample_NormPullGap_data'],
        ini_gap=df_raw['ini_gap'],
        Fmax_pN_BSDB=df_raw['Fmax_pN_BSDB'],
        # ++ add some for energy integration
        pull_data=df_raw['pull_data'],
        forc_data=df_raw['forc_data'],
        Int_Smo_ForcPull=df_raw['Int_Ene'],
    )
    # ++ add new keys on energy
    
    # screen using AA length
    print('a. screen using sequence length...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['seq_len']>max_AASeq_len-2].index, 
        inplace = True
    )
    protein_df.drop(
        protein_df[protein_df['seq_len'] <min_AASeq_len].index, 
        inplace = True
    )
    protein_df=protein_df.reset_index(drop=True)
    print('used sequences #: ', len(protein_df))
    
    print('b. screen using force values...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['Max_Smo_Force']>max_used_Smo_F].index, 
        inplace = True
    )
    # protein_df.drop(
    #     protein_df[protein_df['seq_len'] <min_AA_len].index, 
    #     inplace = True
    # )
    protein_df=protein_df.reset_index(drop=True)
    print('afterwards, sequences #: ', len(protein_df))
    
    
    
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    sns.distplot(
        protein_df['seq_len'],
        bins=50,kde=False, 
        rug=False,norm_hist=False)
    sns.distplot(
        df_raw['seq_len'],
        bins=50,kde=False, 
        rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('AA length')
    outname = store_path+'CSV_1_AALen_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    #
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    sns.distplot(protein_df['Max_Smo_Force'],kde=True, rug=False,norm_hist=False)
    sns.distplot(df_raw['Max_Smo_Force'],kde=True, rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('Max. Force (pN) from MD')
    outname = store_path+'CSV_2_MaxSmoF_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    
    # re-plot the simplified results of SMD
    print('Check selected in SMD records...')

    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( protein_df )):
        if protein_df['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                protein_df['sample_NormPullGap_data'][ii], 
                protein_df['sample_FORCEpN_data'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            ax0.scatter(
                protein_df['NPullGap_for_MaxSmoF'][ii], 
                protein_df['Max_Smo_Force'][ii], 
            )
        else:
            print(protein_df['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized distance btw pulling ends')
    plt.ylabel('Force (pF)')
    outname = store_path+'CSV_3_Screened_SMD_sim_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    
    # ++
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    sns.distplot(protein_df['Int_Smo_ForcPull'],kde=True, rug=False,norm_hist=False)
    sns.distplot(df_raw['Int_Ene'],kde=True, rug=False,norm_hist=False)
    #
    plt.legend(['Selected','Full recrod'])
    plt.xlabel('Integrated energy (pN*Angstrom) from smoothed MD hist.')
    outname = store_path+'CSV_4_MaxSmoF_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print('Done')
    
    
    return df_raw, protein_df

# ============================================================
# convert csv into df: new for smoothed data of SMD
# 2. convert into dataframe
# ============================================================
def load_data_set_from_df_SMD(
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # prefix=None,
    # protein_df=None,
    # batch_size=1,
    # maxdata=9999999999999,
    # test_ratio=0.2,
    # AA_seq_normfac=22,
    # norm_fac_force=1,
    # max_AA_len=128,
    # ControlKeys=None):
    
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    #
    X_Key = PKeys['X_Key']
    #
    max_AA_len = PKeys['max_AA_seq_len']
    norm_fac_force = PKeys['Xnormfac']
    AA_seq_normfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X'] # not be used
    tokenizer_y = PKeys['tokenizer_y'] # should be none
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    
    maxdata=PKeys['maxdata']
    
    # working
    print("======================================================")
    print("1. work on X data")
    print("======================================================")
    X1 = [] # for short scalar: max_force
    X2 = [] # for long vector: force history
    for i in range(len(protein_df)):
        X1.append( [ protein_df['Max_Smo_Force'][i] ] )
        X2.append( 
            pad_a_np_arr(protein_df['sample_FORCEpN_data'][i],0,max_AA_len)
        )
    X1=np.array(X1)
    X2=np.array(X2)
    print('Max_F shape: ', X1.shape)
    print('SMD_F_Path shape', X2.shape)
    
    if X_Key=='Max_Smo_Force':
        X = X1.copy()
    else:
        X = X2.copy()
    print('Use '+X_Key)
    
    # norm_fac_force = np.max(X1[:])
    print('Normalized factor for the force: ', norm_fac_force)
    # normalization
    # X1 = X1/norm_fac_force
    # X2 = X2/norm_fac_force
    X = X/norm_fac_force
    
    # show it
    fig = plt.figure()
    sns.distplot(
        X1[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,
        axlabel='Normalized Fmax')
    # fig = fig_handle.get_figure()
    plt.ylabel('Counts')
    outname = store_path+'CSV_4_AfterNorm_SMD_Fmax.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print("======================================================")
    print("2. work on Y data")
    print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    # tokenizer_y = None
    if tokenizer_y==None:
        tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_y.fit_on_texts(seqs)
    
    #y_data = tokenizer_y.texts_to_sequences(y_data)
    y_data = tokenizer_y.texts_to_sequences(seqs)

    y_data= sequence.pad_sequences(
        y_data,  maxlen=max_AA_len, 
        padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', bins=np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 21)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    print ("#################################")
    print ("DICTIONARY y_data")
    dictt=tokenizer_y.get_config()
    print (dictt)
    num_words = len(tokenizer_y.word_index) + 1
    print ("################## y max token: ",num_words )
    
    #revere
    print ("TEST REVERSE: ")
    y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
    for iii in range (len(y_data_reversed)):
        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # train_dataset = RegressionDataset_Three(
    #     torch.from_numpy(X1_train).float(), 
    #     torch.from_numpy(X2_train).float(), 
    #     torch.from_numpy(y_train).float()/AA_seq_normfac
    # ) #/ynormfac
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), # already normalized
        torch.from_numpy(y_train).float()/AA_seq_normfac
    ) #/ynormfac
    
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), # already normalized
        torch.from_numpy(y_test).float()/AA_seq_normfac
    ) #/ynormfac
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    
    return train_loader, train_loader_noshuffle, \
        test_loader, tokenizer_y, tokenizer_X

# ============================================================
#
# ============================================================
def load_data_set_from_df_SMD_pLM(
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # \\\
    # input: ForcePath as the input sequence: tokenized x_data 
    # output: AASeq
    # ///
    
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    #
    X_Key = PKeys['X_Key']
    #
    max_AA_len = PKeys['max_AA_seq_len']
    norm_fac_force = PKeys['Xnormfac']
    AA_seq_normfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X'] # not be used
    tokenizer_y = PKeys['tokenizer_y'] # should be none
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    maxdata=PKeys['maxdata']
    # ++ for pLM
    
    # working
    print("======================================================")
    print("1. work on X data: Normalized ForcePath")
    print("======================================================")
    X1 = [] # for short scalar: max_force
    X2 = [] # for long vector: force history
    # note, pad_a_np_arr_esm() is designed for ESM: one padding is added at the 0th position
    for i in range(len(protein_df)):
        X1.append( [ protein_df['Max_Smo_Force'][i] ] )
        # X2.append( 
        #     pad_a_np_arr_esm(protein_df['sample_FORCEpN_data'][i],0,max_AA_len)
        # )
        X2.append( 
            pad_a_np_arr_esm(protein_df['sample_FORCE_data'][i],0,max_AA_len)
        )
    X1=np.array(X1)
    X2=np.array(X2)
    print('Max_F shape: ', X1.shape)
    print('SMD_F_Path shape', X2.shape)
    
    if X_Key=='Max_Smo_Force':
        X = X1.copy()
    else:
        X = X2.copy()
    print('Use '+X_Key)
    
    # norm_fac_force = np.max(X1[:])
    print('Normalized factor for the force: ', norm_fac_force)
    # normalization
    # X1 = X1/norm_fac_force
    # X2 = X2/norm_fac_force
    X = X/norm_fac_force
    
    print("tokenizer_X=None")
    # show it
    fig = plt.figure()
    sns.distplot(
        X1[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,
        axlabel='Fmax')
    # fig = fig_handle.get_figure()
    plt.ylabel('Counts')
    outname = store_path+'CSV_4_AfterNorm_SMD_Fmax.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print("======================================================")
    print("2. work on Y data: AA Sequence")
    print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    # ++ for pLM: esm
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("pLM model: ", PKeys['ESM-2_Model'])
    
    if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
        # print('Debug block')
        # embed dim: 1280
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
        # embed dim: 480
        esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t36_3B_UR50D':
        # embed dim: 2560
        esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t30_150M_UR50D':
        # embed dim: 640
        esm_model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    else:
        print("protein language model is not defined.")
        # pass
    # for check
    print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
    print("# of tokens in AA alphabet: ", len_toks)
    # need to save 2 positions for <cls> and <eos>
    esm_batch_converter = esm_alphabet.get_batch_converter(
        truncation_seq_length=PKeys['max_AA_seq_len']-2
    )
    esm_model.eval()  # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # add dummy labels
    seqs_ext=[]
    for i in range(len(seqs)):
        seqs_ext.append(
            (" ", seqs[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    print ("y_data.dim: ", y_data.dtype)   
    
    
#     # --
#     # tokenizer_y = None
#     if tokenizer_y==None:
#         tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_y.fit_on_texts(seqs)
    
#     #y_data = tokenizer_y.texts_to_sequences(y_data)
#     y_data = tokenizer_y.texts_to_sequences(seqs)

#     y_data= sequence.pad_sequences(
#         y_data,  maxlen=max_AA_len, 
#         padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', 
        bins=np.array([i-0.5 for i in range(0,33+3,1)]), # np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # -----------------------------------------------------------
    # print ("#################################")
    # print ("DICTIONARY y_data")
    # dictt=tokenizer_y.get_config()
    # print (dictt)
    # num_words = len(tokenizer_y.word_index) + 1
    # print ("################## y max token: ",num_words )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    
    #revere
    print ("TEST REVERSE: ")
    
#     # --------------------------------------------------------------
#     y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
#     for iii in range (len(y_data_reversed)):
#         y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # assume y_data is reversiable
    y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
    
     
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
#     # -----------------------------------------------------------
#     train_dataset = RegressionDataset(
#         torch.from_numpy(X_train).float(), # already normalized
#         torch.from_numpy(y_train).float()/AA_seq_normfac
#     ) #/ynormfac
    
#     test_dataset = RegressionDataset(
#         torch.from_numpy(X_test).float(), # already normalized
#         torch.from_numpy(y_test).float()/AA_seq_normfac
#     ) #/ynormfac

    # ++ for esm
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), # already normalized
        y_train,
    )
    
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), # already normalized
        y_test,
    )
    
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    
    return train_loader, train_loader_noshuffle, \
        test_loader, tokenizer_y, tokenizer_X

# ============================================================
# forward predictor: From AA to ForcPath
# ============================================================
def load_data_set_from_df_from_pLM_to_SMD(
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # \\\
    # input:  AASeq
    # output: ForcePath as the input sequence: tokenized x_data
    # ///
    
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    #
    X_Key = PKeys['X_Key']
    #
    max_AA_len = PKeys['max_AA_seq_len']
    # # --
    # norm_fac_force = PKeys['Xnormfac']
    # AA_seq_normfac = PKeys['ynormfac']
    # ++
    norm_fac_force = PKeys['ynormfac']
    AA_seq_normfac = PKeys['Xnormfac']
    #
    tokenizer_X = PKeys['tokenizer_X'] # not be used
    tokenizer_y = PKeys['tokenizer_y'] # should be none
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    maxdata=PKeys['maxdata']
    # ++ for pLM
    
    # working
    print("======================================================")
    # print("1. work on X data: Normalized ForcePath")
    print("1. work on Y data: Normalized ForcePath")
    print("======================================================")
    X1 = [] # for short scalar: max_force
    X2 = [] # for long vector: force history
    # note, pad_a_np_arr_esm() is designed for ESM: one padding is added at the 0th position
    for i in range(len(protein_df)):
        X1.append( [ protein_df['Max_Smo_Force'][i] ] )
        # X2.append( 
        #     pad_a_np_arr_esm(protein_df['sample_FORCEpN_data'][i],0,max_AA_len)
        # )
        X2.append( 
            pad_a_np_arr_esm(protein_df['sample_FORCE_data'][i],0,max_AA_len)
        )
    X1=np.array(X1)
    X2=np.array(X2)
    print('Max_F shape: ', X1.shape)
    print('SMD_F_Path shape', X2.shape)
    
    if X_Key=='Max_Smo_Force':
        X = X1.copy()
    else:
        X = X2.copy()
    print('Use '+X_Key)
    
    # norm_fac_force = np.max(X1[:])
    print('Normalized factor for the force: ', norm_fac_force)
    # normalization
    # X1 = X1/norm_fac_force
    # X2 = X2/norm_fac_force
    X = X/norm_fac_force
    
    print("tokenizer_X=None")
    # show it
    fig = plt.figure()
    sns.distplot(
        X1[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,
        axlabel='Fmax')
    # fig = fig_handle.get_figure()
    plt.ylabel('Counts')
    outname = store_path+'CSV_4_AfterNorm_SMD_Fmax.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print("======================================================")
    print("2. work on Y data: AA Sequence")
    print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    if PKeys['ESM-2_Model']=='trivial':
        print("Plain tokenizer of AA sequence is used...")
        # tokenizer_y = None
        if tokenizer_y==None:
            tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
            tokenizer_y.fit_on_texts(seqs)

        #y_data = tokenizer_y.texts_to_sequences(y_data)
        y_data = tokenizer_y.texts_to_sequences(seqs)
        # adjust the padding stratage to accomdate the esm data format
        # task: AAA -> 0aaa0 (add one 0 at the beginning)
        # # --
        # y_data= sequence.pad_sequences(
        #     y_data,  maxlen=max_AA_len, 
        #     padding='post', truncating='post')
        # ++
        y_data= sequence.pad_sequences(
            y_data,  maxlen=max_AA_len-1, 
            padding='post', truncating='post',
            value=0.0,
        )
        # add one 0 at the begining
        y_data= sequence.pad_sequences(
            y_data, maxlen=max_AA_len,
            padding='pre', truncating='pre',
            value=0.0,
        )
        
        len_toks = len(tokenizer_y.word_index) + 1
        
    else:
        # ++ for pLM: esm
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print("pLM model: ", PKeys['ESM-2_Model'])

        if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
            # print('Debug block')
            # embed dim: 1280
            esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
        elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
            # embed dim: 480
            esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
        elif PKeys['ESM-2_Model']=='esm2_t36_3B_UR50D':
            # embed dim: 2560
            esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            len_toks=len(esm_alphabet.all_toks)
        elif PKeys['ESM-2_Model']=='esm2_t30_150M_UR50D':
            # embed dim: 640
            esm_model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
        else:
            print("protein language model is not defined.")
            # pass
        # for check
        print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
        print("# of tokens in AA alphabet: ", len_toks)
        # need to save 2 positions for <cls> and <eos>
        esm_batch_converter = esm_alphabet.get_batch_converter(
            truncation_seq_length=PKeys['max_AA_seq_len']-2
        )
        esm_model.eval()  # disables dropout for deterministic results
        # prepare seqs for the "esm_batch_converter..."
        # add dummy labels
        seqs_ext=[]
        for i in range(len(seqs)):
            seqs_ext.append(
                (" ", seqs[i])
            )
        # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
        _, y_strs, y_data = esm_batch_converter(seqs_ext)
        y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
        # 
        # NEED to check the size of y_data
        # need to dealwith if y_data are only shorter sequences
        # need to add padding with a value, int (1)
        current_seq_len = y_data.shape[1]
        print("current seq batch len: ", current_seq_len)
        missing_num_pad = PKeys['max_AA_seq_len']-current_seq_len
        if missing_num_pad>0:
            print("extra padding is added to match the target seq input length...")
            # padding is needed
            y_data = F.pad(
                y_data,
                (0, missing_num_pad),
                "constant", esm_alphabet.padding_idx
            )
        else:
            print("No extra padding is needed")
        
    
    # ----------------------------------------------------------------------------------
    # print(batch_tokens.shape)
    print ("y_data.dim: ", y_data.shape)  
    print ("y_data.type: ", y_data.type)  
    
#     # --
#     # tokenizer_y = None
#     if tokenizer_y==None:
#         tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_y.fit_on_texts(seqs)
    
#     #y_data = tokenizer_y.texts_to_sequences(y_data)
#     y_data = tokenizer_y.texts_to_sequences(seqs)

#     y_data= sequence.pad_sequences(
#         y_data,  maxlen=max_AA_len, 
#         padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', 
        bins=np.array([i-0.5 for i in range(0,33+3,1)]), # np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    
    # -----------------------------------------------------------
    # print ("#################################")
    # print ("DICTIONARY y_data")
    # dictt=tokenizer_y.get_config()
    # print (dictt)
    # num_words = len(tokenizer_y.word_index) + 1
    # print ("################## y max token: ",num_words )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    
    #revere
    print ("TEST REVERSE: ")
    
    if PKeys['ESM-2_Model']=='trivial':
        # --------------------------------------------------------------
        y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
        for iii in range (len(y_data_reversed)):
            y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
    else:
        # for ESM models
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # assume y_data is reversiable
        y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
    
     
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # for the forward predictor, we SWITCH X and Y here
    if PKeys['ESM-2_Model']=='trivial':
        #
        train_dataset = RegressionDataset(
            torch.from_numpy(y_train).float()/AA_seq_normfac,
            torch.from_numpy(X_train).float(), # already normalized
        )
        #
        test_dataset = RegressionDataset(
            torch.from_numpy(y_test).float()/AA_seq_normfac,
            torch.from_numpy(X_test).float(), # already normalized
        ) #/ynormfac
    else:
        # for ESM
        train_dataset = RegressionDataset(
            y_train,
            torch.from_numpy(X_train).float(), # already normalized
        )
        #
        test_dataset = RegressionDataset(
            y_test,
            torch.from_numpy(X_test).float(), # already normalized
        )
        
# # --    
# #     # -----------------------------------------------------------
# #     train_dataset = RegressionDataset(
# #         torch.from_numpy(X_train).float(), # already normalized
# #         torch.from_numpy(y_train).float()/AA_seq_normfac
# #     ) #/ynormfac
    
# #     test_dataset = RegressionDataset(
# #         torch.from_numpy(X_test).float(), # already normalized
# #         torch.from_numpy(y_test).float()/AA_seq_normfac
# #     ) #/ynormfac

#     # ++ for esm
#     train_dataset = RegressionDataset(
#         torch.from_numpy(X_train).float(), # already normalized
#         y_train,
#     )
    
#     test_dataset = RegressionDataset(
#         torch.from_numpy(X_test).float(), # already normalized
#         y_test,
#     )
    
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    # --
    # return train_loader, train_loader_noshuffle, \
    #     test_loader, tokenizer_y, tokenizer_X
    # ++
    # switch X and Y for foward predictor
    return train_loader, train_loader_noshuffle, \
        test_loader, tokenizer_X, tokenizer_y



# ============================================================
# convert csv into df: new
# ============================================================
def load_data_set_SS_InSeqToOuSeq(
    file_path,
    PKeys=None,
    CKeys=None,
):
    # unload the parameters
    min_AASeq_len = PKeys['min_AA_seq_len']
    max_AASeq_len = PKeys['max_AA_seq_len']
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    Xnormfac = PKeys['Xnormfac']
    ynormfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X']
    tokenizer_y = PKeys['tokenizer_y']
    maxdata=PKeys['maxdata']
    # add some
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    
    
    # working
    protein_df=pd.read_csv(file_path)
    protein_df.describe()
    # 1. check the nulls
    df_isnull = pd.DataFrame(
        round(
            (protein_df.isnull().sum().sort_values(ascending=False)/protein_df.shape[0])*100,
            1
        )
    ).reset_index()
    df_isnull.columns = ['Columns', '% of Missing Data']
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    df_isnull
    # 2. screen the df using seq_len 
    # add Seq_Len if not exists
    protein_df['Seq_Len'] = protein_df.apply(lambda x: len(x['Seq']), axis=1)
     
    protein_df.drop(protein_df[protein_df['Seq_Len'] >max_AASeq_len-2].index, inplace = True)
    protein_df.drop(protein_df[protein_df['Seq_Len'] <min_AASeq_len].index, inplace = True)
    protein_df=protein_df.reset_index(drop=True)
    
    seqs = protein_df.Sequence.values
    
    test_seqs = seqs[:1]
     
    lengths = [len(s) for s in seqs]
    
    print('After the screening using seq_len: ')
    print(protein_df.shape)
    print(protein_df.head(6))
     
    min_length_measured =   min (lengths)
    max_length_measured =   max (lengths)
     
    fig_handle =sns.distplot(
        lengths,  bins=25,
        kde=False, rug=False,
        norm_hist=True,
        axlabel='Length')
    outname=store_path+'0_DataSet_AA_Len_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # 3. process X: generate tokenizer
    X_v = protein_df.Secstructure.values
    if tokenizer_X==None:
        tokenizer_X = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_X.fit_on_texts(X_v)
        
    X = tokenizer_X.texts_to_sequences(X_v)
    X = sequence.pad_sequences(
        X,  maxlen=max_AASeq_len, 
        padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'SecStr code': np.array(X).flatten()}),
        x='SecStr code', bins=np.array([i-0.5 for i in range(0,8+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 9)
    # fig_handle.set_ylim(0, 400000)
    outname=store_path+'1_DataSet_SecStrCode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    print ("#################################")
    print ("DICTIONARY X")
    dictt=tokenizer_X.get_config()
    print (dictt)
    num_wordsX = len(tokenizer_X.word_index) + 1

    print ("################## X max token: ",num_wordsX )

    X=np.array(X)
    print ("sample X data", X[0])
    
    
    # 4. y: AA sequences
    seqs = protein_df.Sequence.values
            
    #create and fit tokenizer for AA sequences
    if tokenizer_y==None:
        tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_y.fit_on_texts(seqs)
        
    y_data = tokenizer_y.texts_to_sequences(seqs)
    y_data= sequence.pad_sequences(
        y_data,  maxlen=max_AASeq_len, 
        padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', bins=np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 21)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'2_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    print ("#################################")
    print ("DICTIONARY y_data")
    dictt=tokenizer_y.get_config()
    print (dictt)
    num_words = len(tokenizer_y.word_index) + 1
    print ("################## y max token: ",num_words )
    
    
    # 5. CHeck the tokenizers:
    print ("y data shape: ", y_data.shape)
    print ("X data shape: ", X.shape)
    
    #revere
    print ("TEST REVERSE: ")

    y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
    for iii in range (len(y_data_reversed)):
        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    print ("Element 0", y_data_reversed[0])

    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )

    # cutting back if there are too many data points
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes: ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # do normalizatoin: training set
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float()/Xnormfac, 
        torch.from_numpy(y_train).float()/ynormfac) #/ynormfac)
    
    fig_handle = sns.distplot(
        torch.from_numpy(y_train.flatten()),
        bins=42,kde=False, 
        rug=False,norm_hist=False,axlabel='y labels')
    fig = fig_handle.get_figure()
    outname=store_path+'3_DataSet_Norm_YTrain_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        torch.from_numpy(X_train.flatten()),
        bins=25,kde=False, 
        rug=False,norm_hist=False,axlabel='X labels')
    fig = fig_handle.get_figure()
    outname=store_path+'3_DataSet_Norm_XTrain_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # do normalizatoin: training set
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float()/Xnormfac, 
        torch.from_numpy(y_test).float()/ynormfac)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y , tokenizer_X
    
# ============================================================
# convert csv into df: new
# ============================================================
from PD_pLMProbXDiff.UtilityPack import decode_one_ems_token_rec,decode_many_ems_token_rec
# # have been moved into UtilityPack
# def decode_one_ems_token_rec(this_token, esm_alphabet):
#     # print( (this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0] )
#     # print( (this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0] )
#     # print( (this_token==100).nonzero(as_tuple=True)[0]==None )

#     id_b=(this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0]
#     id_e=(this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0]
#     if len(id_e)==0:
#         # no ending for this one
#         id_e=len(this_token)

#     this_seq = []
#     for ii in range(id_b+1,id_e,1):
#         this_seq.append(
#             esm_alphabet.get_tok(this_token[ii])
#         )
        
#     this_seq = "".join(this_seq)

#     # print(this_seq)    
#     # print(len(this_seq))
#     # # print(this_token[id_b+1:id_e]) 
#     return this_seq

# def decode_many_ems_token_rec(batch_tokens, esm_alphabet):
#     rev_y_seq = []
#     for jj in range(len(batch_tokens)):
#         # do for one seq: this_seq
#         this_seq = decode_one_ems_token_rec(
#             batch_tokens[jj], esm_alphabet
#             )
#         rev_y_seq.append(this_seq)
#     return rev_y_seq
# # ...............................................................

def load_data_set_SS_InSeqToOuSeq_pLM(
    file_path,
    PKeys=None,
    CKeys=None,
):
    #\\\ To be implemented
    # input:  SecStr seq
    # output: AASeq
    # for training purpose: output will be embedding presentation
    #///
    # unload the parameters
    min_AASeq_len = PKeys['min_AA_seq_len']
    max_AASeq_len = PKeys['max_AA_seq_len']
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    Xnormfac = PKeys['Xnormfac']
    ynormfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X']
    tokenizer_y = PKeys['tokenizer_y']
    maxdata=PKeys['maxdata']
    # add some
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    
    
    # working
    protein_df=pd.read_csv(file_path)
    protein_df.describe()
    # 1. check the nulls
    df_isnull = pd.DataFrame(
        round(
            (protein_df.isnull().sum().sort_values(ascending=False)/protein_df.shape[0])*100,
            1
        )
    ).reset_index()
    df_isnull.columns = ['Columns', '% of Missing Data']
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    df_isnull
    # 2. screen the df using seq_len 
    # add Seq_Len if not exists
    protein_df['Seq_Len'] = protein_df.apply(lambda x: len(x['Seq']), axis=1)
    # save two for beginning and ending 
    protein_df.drop(protein_df[protein_df['Seq_Len'] >max_AASeq_len-2].index, inplace = True)
    protein_df.drop(protein_df[protein_df['Seq_Len'] <min_AASeq_len].index, inplace = True)
    protein_df=protein_df.reset_index(drop=True)
    
    seqs = protein_df.Sequence.values
    
    test_seqs = seqs[:1]
     
    lengths = [len(s) for s in seqs]
    
    print('After the screening using seq_len: ')
    print(protein_df.shape)
    print(protein_df.head(6))
     
    min_length_measured =   min (lengths)
    max_length_measured =   max (lengths)
     
    fig_handle =sns.distplot(
        lengths,  bins=25,
        kde=False, rug=False,
        norm_hist=True,
        axlabel='Length')
    outname=store_path+'0_DataSet_AA_Len_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # 3. process X: generate tokenizer
    X_v = protein_df.Secstructure.values
    if tokenizer_X==None:
        tokenizer_X = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_X.fit_on_texts(X_v)
        
    X = tokenizer_X.texts_to_sequences(X_v)
    # print(type(X))
    # print(X[0])
    # # print(X[0].insert(0,0))
    # print([0]+X[0])
    # print(X)
    # by default, 0 is used as the padding
    # for pLM, need to put one padding at the beginning
    X1=[]
    for this_X in X:
        X1.append(
            [0]+this_X
        )
    # print(type(X1))
    print(X1[0])
    # pad the endding
    X = sequence.pad_sequences(
        X1,  maxlen=max_AASeq_len, 
        padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'SecStr code': np.array(X).flatten()}),
        x='SecStr code', bins=np.array([i-0.5 for i in range(0,8+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 9)
    # fig_handle.set_ylim(0, 400000)
    outname=store_path+'1_DataSet_SecStrCode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    print ("#################################")
    print ("DICTIONARY X")
    dictt=tokenizer_X.get_config()
    print (dictt)
    num_wordsX = len(tokenizer_X.word_index) + 1

    print ("################## X max token: ",num_wordsX )

    X=np.array(X)
    print ("sample X data", X[0])
    
    
    # 4. y: AA sequences
    seqs = protein_df.Sequence.values
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
        # print('Debug block')
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
        esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    else:
        pass
    
    # for check
    print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
    print("# of tokens in AA alphabet: ", len_toks)

    esm_batch_converter = esm_alphabet.get_batch_converter(
        truncation_seq_length=PKeys['max_AA_seq_len']-2
    )
    esm_model.eval()  # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # add dummy labels
    seqs_ext=[]
    for i in range(len(seqs)):
        seqs_ext.append(
            (" ", seqs[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    
    # # ------------------------------------------------------------------        
    # #create and fit tokenizer for AA sequences
    # if tokenizer_y==None:
    #     tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
    #     tokenizer_y.fit_on_texts(seqs)
        
    # y_data = tokenizer_y.texts_to_sequences(seqs)
    # y_data= sequence.pad_sequences(
    #     y_data,  maxlen=max_AASeq_len, 
    #     padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA token (esm)': np.array(y_data).flatten()}),
        x='AA token (esm)', bins=np.array([i-0.5 for i in range(0,33+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    # fig_handle.set_xlim(-1, 21)
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'2_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # # ---------------------------------------------------------------
    # print ("#################################")
    # print ("DICTIONARY y_data")
    # dictt=tokenizer_y.get_config()
    # print (dictt)
    # num_words = len(tokenizer_y.word_index) + 1
    # print ("################## y max token: ",num_words )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    # ++
    # change the shape of the torch: adding an channel dimension
    # for AA_seq, here change it into AA_tokens, whose dim: (batch, seq_len)
    # # y_data: torch.tensor
    # y_data = torch.unsqueeze(y_data, dim=1)
    # X: numpy array
    # X dim: [batch, channel=1, seq_len]
    # X = np.expand_dims(X, axis=1)
    
    # 5. CHeck the tokenizers:
    print ("y data shape: ", y_data.shape) # (Sample#, max_AA_length)
    print ("X data shape: ", X.shape)      # (Sample#, max_AA_length)
    
    #revere
    print ("TEST REVERSE: ")

    # # ----------------------------------------------------------------
    # y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    # for iii in range (len(y_data_reversed)):
    #     y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
        
    print ("Element 0", y_data_reversed[0])

    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )

    # cutting back if there are too many data points
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes: ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # do normalizatoin: training set
    # # ------------------------------------------------------------
    # train_dataset = RegressionDataset(
    #     torch.from_numpy(X_train).float()/Xnormfac, 
    #     torch.from_numpy(y_train).float()/ynormfac) #/ynormfac)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float()/Xnormfac, 
        y_train) # which is torch.tensor already; No normalization needed
    
    # # ------------------------------------------------------------
    # fig_handle = sns.distplot(
    #     torch.from_numpy(y_train.flatten()),
    #     bins=42,kde=False, 
    #     rug=False,norm_hist=False,axlabel='y labels')
    # fig = fig_handle.get_figure()
    # outname=store_path+'3_DataSet_Norm_YTrain_dist.jpg'
    # if IF_SaveFig==1:
    #     plt.savefig(outname, dpi=200)
    # else:
    #     plt.show()
    # plt.close()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fig_handle = sns.distplot(
    #     torch.flatten( y_train),
    #     bins=33,kde=False, 
    #     rug=False,norm_hist=False,axlabel='y labels')
    # fig = fig_handle.get_figure()
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA token (esm)': np.array(y_train).flatten()}),
        x='AA token (esm)', bins=np.array([i-0.5 for i in range(0,33+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    # fig_handle.set_xlim(-1, 21)
    fig_handle.set_xlim(-1, 33+1)
    outname=store_path+'3_DataSet_Norm_YTrain_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    
    
    # --------------------------------------------------------------
    # fig_handle = sns.distplot(
    #     torch.from_numpy(X_train.flatten()),
    #     bins=25,kde=False, 
    #     rug=False,norm_hist=False,axlabel='X labels')
    # fig = fig_handle.get_figure()
    fig_handle = sns.histplot(
        data=pd.DataFrame({'SecStr code': np.array(X_train).flatten()}),
        x='SecStr code', bins=np.array([i-0.5 for i in range(0,8+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 9)
    outname=store_path+'3_DataSet_Norm_XTrain_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fig_handle = sns.distplot(
    #     torch.flatten(X_train),
    #     bins=25,kde=False, 
    #     rug=False,norm_hist=False,axlabel='X labels')
    # fig = fig_handle.get_figure()
    # outname=store_path+'3_DataSet_Norm_XTrain_dist.jpg'
    # if IF_SaveFig==1:
    #     plt.savefig(outname, dpi=200)
    # else:
    #     plt.show()
    # plt.close()
    
    # do normalizatoin: training set
    # # -----------------------------------------------------
    # test_dataset = RegressionDataset(
    #     torch.from_numpy(X_test).float()/Xnormfac, 
    #     torch.from_numpy(y_test).float()/ynormfac)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float()/Xnormfac, 
        y_test)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y , tokenizer_X
    

# ============================================================
# dataloader for SecStr-ModelA
# ============================================================
def load_data_set_seq2seq_SecStr_ModelA (
    file_path, 
    PKeys=None,
    CKeys=None,
):
    # unload parameters:
    min_length=PKeys['min_AA_seq_len'] 
    max_length=PKeys['max_AA_seq_len'] 
    batch_size_=PKeys['batch_size']
    # output_dim=1, 
    maxdata=PKeys['maxdata'] # 9999999999999,
    tokenizer_y=PKeys['tokenizer_y'] # will be updated in this function 
    split=PKeys['testset_ratio']
    ynormfac=PKeys['ynormfac']
    # dummy ones
    Xnormfac=PKeys['Xnormfac']
    tokenizer_X=PKeys['tokenizer_X']
    # add some for convenience
    IF_SaveFig = CKeys['SlientRun']
    save_dir = PKeys['data_dir']
    
    
    # min_length_measured=0
    # max_length_measured=999999
    protein_df=pd.read_csv(file_path)
    protein_df

    protein_df.describe()
     
    df_isnull = pd.DataFrame(round((protein_df.isnull().sum().sort_values(ascending=False)/protein_df.shape[0])*100,1)).reset_index()
    df_isnull.columns = ['Columns', '% of Missing Data']
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    df_isnull
    
    # add Seq_Len if it not exist
    protein_df['Seq_Len'] = protein_df.apply(lambda x: len(x['Seq']), axis=1)
    
    print("Screen the sequence length...")
    protein_df.drop(protein_df[protein_df['Seq_Len'] >max_length-2].index, inplace = True)
    protein_df.drop(protein_df[protein_df['Seq_Len'] <min_length].index, inplace = True)

    print("# of data point: ", protein_df.shape)
    print(protein_df.head(6))
    protein_df=protein_df.reset_index(drop=True)
    
    seqs = protein_df.Sequence.values
    
    test_seqs = seqs[:1]
  
    lengths = [len(s) for s in seqs]

    print(protein_df.shape)
    print(protein_df.head(6))
    
    min_length_measured =   min (lengths)
    max_length_measured =   max (lengths)
    
    print ("Measured seq length: min and max")
    print (min_length_measured,max_length_measured)
    
    
    fig_handle = sns.distplot(
        lengths,
        bins=50,
        kde=False, 
        rug=False,
        norm_hist=False,
        axlabel='Length'
    )
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_0_measured_AA_len_Dist.jpg'
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    # check the keys 
    print("keys in df: ", protein_df.keys())
    
    #INPUT - X
    # No need for normalizeation and tokenization
    # fake ones will be provided
    X=[]
    for i in range  (len(seqs)):
        X.append([protein_df['AH'][i],protein_df['BS'][i],protein_df['T'][i],
                  protein_df['UNSTRUCTURED'][i],protein_df['BETABRIDGE'][i],
                  protein_df['310HELIX'][i],protein_df['PIHELIX'][i],protein_df['BEND'][i]
                 ])
           
    X=np.array(X)
    print ("sample X data", X[0])
    
    fig_handle = sns.distplot(
        X[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='AH')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_1_AH_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,1],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BS')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_2_BS_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,2],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='T')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_3_T_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,3],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='UNSTRUCTURED')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_4_UNSTRUCTURED_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,4],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BETABRIDGE')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_5_BETABRIDGE_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    
    fig_handle = sns.distplot(
        X[:,5],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='310HELIX')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_6_310HELIX_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,6],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='PIHELIX')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_7_PIHELIX_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,7],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BEND')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_8_BEND_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    # on Y:
    seqs = protein_df.Sequence.values
            
    #create and fit tokenizer for AA sequences
    if tokenizer_y==None:
        tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_y.fit_on_texts(seqs)
        
    y_data = tokenizer_y.texts_to_sequences(seqs)
    
    y_data= sequence.pad_sequences(y_data,  maxlen=max_length, padding='post', truncating='post')  
 
    print ("#################################")
    print ("DICTIONARY y_data")
    dictt=tokenizer_y.get_config()
    print (dictt)
    num_words = len(tokenizer_y.word_index) + 1

    print ("################## max token: ",num_words )

    #revere
    print ("TEST REVERSE: ")
    print ("y data shape: ", y_data.shape)
    y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
    for iii in range (len(y_data_reversed)):
        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    print ("Element 0", y_data_reversed[0])
    
    print ("Number of y samples",len (y_data_reversed) )
    print ("Original: ", y_data[:3,:])
    print ("Original Seq      : ", seqs[:3])

    print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])

    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes, X and y:\n ", X.shape, y_data.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data , 
        test_size=split,
        random_state=235
    )
        
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).float()/ynormfac
    ) #/ynormfac)
    
    
    # # ---------------------------------------
    # fig_handle = sns.distplot(torch.from_numpy(y_train)*ynormfac,bins=25,kde=False, rug=False,norm_hist=False,axlabel='y: AA labels')
    # fig = fig_handle.get_figure()
    # plt.show()
    # +++++++++++++++++++++++++++++++++++++++++++++
    fig_handle = sns.distplot(
        torch.from_numpy(y_train.flatten()),
        bins=42,kde=False, 
        rug=False,norm_hist=False,axlabel='y: AA labels')
    fig = fig_handle.get_figure()
    outname=save_dir+'CSV_9_YTrain_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).float()/ynormfac
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size_, 
        shuffle=True
    )
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size_, 
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size_
    )
    
    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y, tokenizer_X

# ============================================================
# dataloader for SecStr-ModelA
# extend for ESM pLM model
# ============================================================
def load_data_set_seq2seq_SecStr_ModelA_pLM (
    file_path, 
    PKeys=None,
    CKeys=None,
):
    # \\\ 
    # input:  SecStr text vector
    # output: AASeq
    # for training purpose: output will be in embedding presentation
    # ///
    #
    # unload parameters:
    min_length=PKeys['min_AA_seq_len'] 
    max_length=PKeys['max_AA_seq_len'] 
    batch_size_=PKeys['batch_size']
    # output_dim=1, 
    maxdata=PKeys['maxdata'] # 9999999999999,
    tokenizer_y=PKeys['tokenizer_y'] # will be updated in this function 
    split=PKeys['testset_ratio']
    ynormfac=PKeys['ynormfac']
    # dummy ones
    Xnormfac=PKeys['Xnormfac']
    tokenizer_X=PKeys['tokenizer_X']
    # add some for convenience
    IF_SaveFig = CKeys['SlientRun']
    save_dir = PKeys['data_dir']
    
    
    # min_length_measured=0
    # max_length_measured=999999
    protein_df=pd.read_csv(file_path)
    protein_df

    protein_df.describe()
     
    df_isnull = pd.DataFrame(round((protein_df.isnull().sum().sort_values(ascending=False)/protein_df.shape[0])*100,1)).reset_index()
    df_isnull.columns = ['Columns', '% of Missing Data']
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    df_isnull
    
    # add Seq_Len if it not exist
    protein_df['Seq_Len'] = protein_df.apply(lambda x: len(x['Seq']), axis=1)
    
    print("Screen the sequence length...")
    protein_df.drop(protein_df[protein_df['Seq_Len'] >max_length-2].index, inplace = True)
    protein_df.drop(protein_df[protein_df['Seq_Len'] <min_length].index, inplace = True)

    print("# of data point: ", protein_df.shape)
    print(protein_df.head(6))
    protein_df=protein_df.reset_index(drop=True)
    
    seqs = protein_df.Sequence.values
    
    test_seqs = seqs[:1]
  
    lengths = [len(s) for s in seqs]

    print(protein_df.shape)
    print(protein_df.head(6))
    
    min_length_measured =   min (lengths)
    max_length_measured =   max (lengths)
    
    print ("Measured seq length: min and max")
    print (min_length_measured,max_length_measured)
    
    
    fig_handle = sns.distplot(
        lengths,
        bins=50,
        kde=False, 
        rug=False,
        norm_hist=False,
        axlabel='Length'
    )
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_0_measured_AA_len_Dist.jpg'
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    # check the keys 
    print("keys in df: ", protein_df.keys())
    
    #INPUT - X
    # No need for normalizeation and tokenization
    # fake ones will be provided
    X=[]
    for i in range  (len(seqs)):
        X.append([protein_df['AH'][i],protein_df['BS'][i],protein_df['T'][i],
                  protein_df['UNSTRUCTURED'][i],protein_df['BETABRIDGE'][i],
                  protein_df['310HELIX'][i],protein_df['PIHELIX'][i],protein_df['BEND'][i]
                 ])
           
    X=np.array(X)
    print ("X.shape: ", X.shape, "[batch, text_len=8]")
    print ("sample X data", X[0])
    
    fig_handle = sns.distplot(
        X[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='AH')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_1_AH_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,1],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BS')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_2_BS_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,2],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='T')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_3_T_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,3],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='UNSTRUCTURED')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_4_UNSTRUCTURED_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,4],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BETABRIDGE')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_5_BETABRIDGE_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    
    fig_handle = sns.distplot(
        X[:,5],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='310HELIX')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_6_310HELIX_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,6],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='PIHELIX')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_7_PIHELIX_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    fig_handle = sns.distplot(
        X[:,7],bins=50,kde=False, 
        rug=False,norm_hist=False,axlabel='BEND')
    fig = fig_handle.get_figure()
    outname = save_dir+'CSV_8_BEND_Dist.jpg'
    plt.ylabel("Counts")
    plt.savefig(outname, dpi=200)
    if IF_SaveFig==1:
        pass
    else:  
        plt.show()
    plt.close()
    
    # on Y: AA sequences
    seqs = protein_df.Sequence.values
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
        # print('Debug block')
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
        esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    else:
        pass
    
    # for check
    print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
    print("# of tokens in AA alphabet: ", len_toks)

    esm_batch_converter = esm_alphabet.get_batch_converter(
        truncation_seq_length=PKeys['max_AA_seq_len']-2
    )
    esm_model.eval()  # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # add dummy labels
    seqs_ext=[]
    for i in range(len(seqs)):
        seqs_ext.append(
            (" ", seqs[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA token (esm)': np.array(y_data).flatten()}),
        x='AA token (esm)', bins=np.array([i-0.5 for i in range(0,33+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    # fig_handle.set_xlim(-1, 21)
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=save_dir+'CSV_9_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
#     # -----------------------------------------------------------        
#     #create and fit tokenizer for AA sequences
#     if tokenizer_y==None:
#         tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_y.fit_on_texts(seqs)
        
#     y_data = tokenizer_y.texts_to_sequences(seqs)
    
#     y_data= sequence.pad_sequences(y_data,  maxlen=max_length, padding='post', truncating='post')  
 
#     print ("#################################")
#     print ("DICTIONARY y_data")
#     dictt=tokenizer_y.get_config()
#     print (dictt)
#     num_words = len(tokenizer_y.word_index) + 1

#     print ("################## max token: ",num_words )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    # 5. CHeck the tokenizers:
    print ("y data shape: ", y_data.shape) # (Sample#, max_AA_length)
    print ("X data shape: ", X.shape)      # (Sample#, max_AA_length)
    
    #revere
    print ("TEST REVERSE: ")
#     # -------------------------------------------------------
#     print ("y data shape: ", y_data.shape)
#     y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
#     for iii in range (len(y_data_reversed)):
#         y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
#     print ("Element 0", y_data_reversed[0])
    
#     print ("Number of y samples",len (y_data_reversed) )
#     print ("Original: ", y_data[:3,:])
#     print ("Original Seq      : ", seqs[:3])

#     print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])

#     print ("Len 0 as example: ", len (y_data_reversed[0]) )
#     print ("Len 2 as example: ", len (y_data_reversed[2]) )
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
        
    print ("Element 0", y_data_reversed[0])

    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes, X and y:\n ", X.shape, y_data.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data , 
        test_size=split,
        random_state=235
    )
    
    # # -----------------------------------------------
    # train_dataset = RegressionDataset(
    #     torch.from_numpy(X_train).float(), 
    #     torch.from_numpy(y_train).float()/ynormfac
    # ) #/ynormfac)
    # +++++++++++++++++++++++++++++++++++++++++++++++
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), 
        y_train,
    ) #/ynormfac)
    
    
    # # ---------------------------------------
    # fig_handle = sns.distplot(torch.from_numpy(y_train)*ynormfac,bins=25,kde=False, rug=False,norm_hist=False,axlabel='y: AA labels')
    # fig = fig_handle.get_figure()
    # plt.show()
    # # +++++++++++++++++++++++++++++++++++++++++++++
    # fig_handle = sns.distplot(
    #     torch.from_numpy(y_train.flatten()),
    #     bins=42,kde=False, 
    #     rug=False,norm_hist=False,axlabel='y: AA labels')
    # fig = fig_handle.get_figure()
    # outname=save_dir+'CSV_10_YTrain_dist.jpg'
    # if IF_SaveFig==1:
    #     plt.savefig(outname, dpi=200)
    # else:
    #     plt.show()
    # plt.close()
    # ++++++++++++++++++++++++++++++++++++++++++++++
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA token (esm)': np.array(y_train).flatten()}),
        x='AA token (esm)', bins=np.array([i-0.5 for i in range(0,33+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    # fig_handle.set_xlim(-1, 21)
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=save_dir+'CSV_10_TrainSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    
    # # --------------------------------------------------
    # test_dataset = RegressionDataset(
    #     torch.from_numpy(X_test).float(), 
    #     torch.from_numpy(y_test).float()/ynormfac
    # )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), 
        y_test,
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size_, 
        shuffle=True
    )
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size_, 
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size_
    )
    
    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y, tokenizer_X

# ============================================================
# dataloader for SMD-ModelA
# ============================================================
def load_data_set_text2seq_MD_ModelA (
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # prefix=None,
    # protein_df=None,
    # batch_size=1,
    # maxdata=9999999999999,
    # test_ratio=0.2,
    # AA_seq_normfac=22,
    # norm_fac_force=1,
    # max_AA_len=128,
    # ControlKeys=None):
    
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    #
    X_Key = PKeys['X_Key']
    #
    max_AA_len = PKeys['max_AA_seq_len']
    norm_fac_force = PKeys['Xnormfac'][0]
    norm_fac_energ = PKeys['Xnormfac'][1]
    #
    AA_seq_normfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X'] # not be used
    tokenizer_y = PKeys['tokenizer_y'] # should be none
    assert tokenizer_X==None, "tokenizer_X should be None"
    assert tokenizer_y==None, "Check tokenizer_y"
    
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    
    maxdata=PKeys['maxdata']
    
    # working
    print("======================================================")
    print("1. work on X data")
    print("======================================================")
    # +
    X = [] # for short scalar: max_force
    for i in range(len(protein_df)):
        X.append( 
            [protein_df[X_Key[0]][i],
             protein_df[X_Key[1]][i]]
        )
    X=np.array(X)        
    print('Input tokenized X.dim: ', X.shape)
    print('Use ', X_Key)
    # # -
    # X1 = [] # for short scalar: max_force
    # X2 = [] # for long vector: force history
    # for i in range(len(protein_df)):
    #     X1.append( [ protein_df['Max_Smo_Force'][i] ] )
    #     X2.append( 
    #         pad_a_np_arr(protein_df['sample_FORCEpN_data'][i],0,max_AA_len)
    #     )
    # X1=np.array(X1)
    # X2=np.array(X2)
    # print('Max_F shape: ', X1.shape)
    # print('SMD_F_Path shape', X2.shape)
    
    # if X_Key=='Max_Smo_Force':
    #     X = X1.copy()
    # else:
    #     X = X2.copy()
    # print('Use '+X_Key)
    
    # norm_fac_force = np.max(X1[:])
    print("Normalize the X input: ...")
    print('Normalized factor for the force: ', norm_fac_force)
    print('Normalized factor for the energy: ', norm_fac_energ)
    # normalization
    # X1 = X1/norm_fac_force
    # X2 = X2/norm_fac_force
    # X = X/norm_fac_force
    X[:,0]=X[:,0]/norm_fac_force
    X[:,1]=X[:,1]/norm_fac_energ
    
    # show it
    fig = plt.figure()
    sns.distplot(
        X[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,
        # axlabel='Normalized Fmax'
    )
    sns.distplot(
        X[:,1],bins=50,kde=False, 
        rug=False,norm_hist=False,
        # axlabel='Normalized Unfolding Ene.'
    )
    # fig = fig_handle.get_figure()
    plt.legend(['Normalized Fmax','Normalized Unfolding Ene.'])
    plt.ylabel('Counts')
    outname = store_path+'CSV_5_AfterNorm_SMD_FmaxEne.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print("======================================================")
    print("2. work on Y data")
    print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    # tokenizer_y = None
    if tokenizer_y==None:
        tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
        tokenizer_y.fit_on_texts(seqs)
    
    #y_data = tokenizer_y.texts_to_sequences(y_data)
    y_data = tokenizer_y.texts_to_sequences(seqs)

    y_data= sequence.pad_sequences(
        y_data,  maxlen=max_AA_len, 
        padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', bins=np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 21)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    print ("#################################")
    print ("DICTIONARY y_data")
    dictt=tokenizer_y.get_config()
    print (dictt)
    num_words = len(tokenizer_y.word_index) + 1
    print ("################## y max token: ",num_words )
    
    #revere
    print ("TEST REVERSE: ")
    y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
    for iii in range (len(y_data_reversed)):
        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # train_dataset = RegressionDataset_Three(
    #     torch.from_numpy(X1_train).float(), 
    #     torch.from_numpy(X2_train).float(), 
    #     torch.from_numpy(y_train).float()/AA_seq_normfac
    # ) #/ynormfac
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), # already normalized
        torch.from_numpy(y_train).float()/AA_seq_normfac
    ) #/ynormfac
    
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), # already normalized
        torch.from_numpy(y_test).float()/AA_seq_normfac
    ) #/ynormfac
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    
    return train_loader, train_loader_noshuffle, \
        test_loader, tokenizer_y, tokenizer_X

# ============================================================
# dataloader for SMD-ModelA-pLM
# ============================================================
def load_data_set_text2seq_MD_ModelA_pLM (
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # \\\
    # input: FmaxEne text 
    # output: AAseq
    # ///
    #
    # prefix=None,
    # protein_df=None,
    # batch_size=1,
    # maxdata=9999999999999,
    # test_ratio=0.2,
    # AA_seq_normfac=22,
    # norm_fac_force=1,
    # max_AA_len=128,
    # ControlKeys=None):
    
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SlientRun']
    #
    X_Key = PKeys['X_Key']
    #
    max_AA_len = PKeys['max_AA_seq_len']
    norm_fac_force = PKeys['Xnormfac'][0]
    norm_fac_energ = PKeys['Xnormfac'][1]
    #
    AA_seq_normfac = PKeys['ynormfac']
    tokenizer_X = PKeys['tokenizer_X'] # not be used
    tokenizer_y = PKeys['tokenizer_y'] # should be none
    assert tokenizer_X==None, "tokenizer_X should be None"
    assert tokenizer_y==None, "Check tokenizer_y"
    
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    
    maxdata=PKeys['maxdata']
    
    # working
    print("======================================================")
    print("1. work on X data")
    print("======================================================")
    # +
    X = [] # for short scalar: max_force
    for i in range(len(protein_df)):
        X.append( 
            [protein_df[X_Key[0]][i],
             protein_df[X_Key[1]][i]]
        )
    X=np.array(X)        
    print('Input tokenized X.dim: ', X.shape)
    print('Use ', X_Key)
    # # -
    # X1 = [] # for short scalar: max_force
    # X2 = [] # for long vector: force history
    # for i in range(len(protein_df)):
    #     X1.append( [ protein_df['Max_Smo_Force'][i] ] )
    #     X2.append( 
    #         pad_a_np_arr(protein_df['sample_FORCEpN_data'][i],0,max_AA_len)
    #     )
    # X1=np.array(X1)
    # X2=np.array(X2)
    # print('Max_F shape: ', X1.shape)
    # print('SMD_F_Path shape', X2.shape)
    
    # if X_Key=='Max_Smo_Force':
    #     X = X1.copy()
    # else:
    #     X = X2.copy()
    # print('Use '+X_Key)
    
    # norm_fac_force = np.max(X1[:])
    print("Normalize the X input: ...")
    print('Normalized factor for the force: ', norm_fac_force)
    print('Normalized factor for the energy: ', norm_fac_energ)
    # normalization
    # X1 = X1/norm_fac_force
    # X2 = X2/norm_fac_force
    # X = X/norm_fac_force
    X[:,0]=X[:,0]/norm_fac_force
    X[:,1]=X[:,1]/norm_fac_energ
    
    # show it
    fig = plt.figure()
    sns.distplot(
        X[:,0],bins=50,kde=False, 
        rug=False,norm_hist=False,
        # axlabel='Normalized Fmax'
    )
    sns.distplot(
        X[:,1],bins=50,kde=False, 
        rug=False,norm_hist=False,
        # axlabel='Normalized Unfolding Ene.'
    )
    # fig = fig_handle.get_figure()
    plt.legend(['Normalized Fmax','Normalized Unfolding Ene.'])
    plt.ylabel('Counts')
    outname = store_path+'CSV_5_AfterNorm_SMD_FmaxEne.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    print("======================================================")
    print("2. work on Y data: AA Sequence with pLM")
    print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    
    # ++ for pLM: esm
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
        # print('Debug block')
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
        esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    else:
        print("protein language model is not defined.")
        # pass
    # for check
    print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
    print("# of tokens in AA alphabet: ", len_toks)
    # need to save 2 positions for <cls> and <eos>
    esm_batch_converter = esm_alphabet.get_batch_converter(
        truncation_seq_length=PKeys['max_AA_seq_len']-2
    )
    esm_model.eval()  # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # add dummy labels
    seqs_ext=[]
    for i in range(len(seqs)):
        seqs_ext.append(
            (" ", seqs[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    print ("y_data: ", y_data.dtype)
    
    
#     # --
#     # tokenizer_y = None
#     if tokenizer_y==None:
#         tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_y.fit_on_texts(seqs)
    
#     #y_data = tokenizer_y.texts_to_sequences(y_data)
#     y_data = tokenizer_y.texts_to_sequences(seqs)

#     y_data= sequence.pad_sequences(
#         y_data,  maxlen=max_AA_len, 
#         padding='post', truncating='post')
    
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', 
        bins=np.array([i-0.5 for i in range(0,33+3,1)]), # np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # # --------------------------------------------------------
    # print ("#################################")
    # print ("DICTIONARY y_data")
    # dictt=tokenizer_y.get_config()
    # print (dictt)
    # num_words = len(tokenizer_y.word_index) + 1
    # print ("################## y max token: ",num_words )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    #revere
    print ("TEST REVERSE: ")
#     # ----------------------------------------------------------------
#     y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
#     for iii in range (len(y_data_reversed)):
#         y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # assume y_data is reversiable
    y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
        
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # 6. Convert into dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X,  y_data, 
        test_size=TestSet_ratio,
        random_state=235)
    
    # train_dataset = RegressionDataset_Three(
    #     torch.from_numpy(X1_train).float(), 
    #     torch.from_numpy(X2_train).float(), 
    #     torch.from_numpy(y_train).float()/AA_seq_normfac
    # ) #/ynormfac
    # # --
    # train_dataset = RegressionDataset(
    #     torch.from_numpy(X_train).float(), # already normalized
    #     torch.from_numpy(y_train).float()/AA_seq_normfac
    # ) #/ynormfac
    # ++ for esm
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), # already normalized
        y_train
    )
    
    # # --
    # test_dataset = RegressionDataset(
    #     torch.from_numpy(X_test).float(), # already normalized
    #     torch.from_numpy(y_test).float()/AA_seq_normfac
    # ) #/ynormfac
    # ++ for esm
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), # already normalized
        y_test,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size)
    
    
    return train_loader, train_loader_noshuffle, \
        test_loader, tokenizer_y, tokenizer_X


# # ============================================================
# # convert csv into df
# # ============================================================
# def load_data_set_SS_seq2seq (
#     file_path,
#     min_length=0, 
#     max_length=100, 
#     batch_size_=4, 
#     output_dim=1, 
#     maxdata=9999999999999,
#     remove_longer=False, 
#     fill_with_repeats=False,
#     repeat_same=True,   
#     tokenizer_y=None, 
#     split=.2,
#     tokenizer_X=None,
#     # ++++++++++++++++++++++++
#     # add some
#     Xnormfac=1.,
#     ynormfac=1.,
# ):
 
    
#     min_length_measured=0
#     max_length_measured=999999
#     protein_df=pd.read_csv(file_path)
#     protein_df

#     protein_df.describe()

#     df_isnull = pd.DataFrame(round((protein_df.isnull().sum().sort_values(ascending=False)/protein_df.shape[0])*100,1)).reset_index()
#     df_isnull.columns = ['Columns', '% of Missing Data']
#     df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
#     cm = sns.light_palette("skyblue", as_cmap=True)
#     df_isnull = df_isnull.style.background_gradient(cmap=cm)
#     df_isnull

#     # add Seq_Len if it not exist
#     protein_df['Seq_Len'] = protein_df.apply(lambda x: len(x['Seq']), axis=1)
     
#     protein_df.drop(protein_df[protein_df['Seq_Len'] >max_length-2].index, inplace = True)
#     protein_df.drop(protein_df[protein_df['Seq_Len'] <min_length].index, inplace = True)
 
#     protein_df=protein_df.reset_index(drop=True)
    
#     seqs = protein_df.Sequence.values
    
#     test_seqs = seqs[:1]
     
#     lengths = [len(s) for s in seqs]
    
#     print(protein_df.shape)
#     print(protein_df.head(6))
     
#     min_length_measured =   min (lengths)
#     max_length_measured =   max (lengths)
     
#     fig_handle =sns.distplot(lengths,  bins=25,kde=False, rug=False,norm_hist=True,axlabel='Length')

#     plt.show()
    
#     #INPUT - X  
#     X_v = protein_df.Secstructure.values
#     if tokenizer_X==None:
#         tokenizer_X = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_X.fit_on_texts(X_v)
        
#     X = tokenizer_X.texts_to_sequences(X_v)
    
#     X= sequence.pad_sequences(X,  maxlen=max_length, padding='post', truncating='post') 
    
#     # fig_handle =sns.distplot(np.array(X).flatten(), bins=9,kde=False, rug=False,norm_hist=False,axlabel='Amino acid residue code')
#     # fig = fig_handle.get_figure()
#     # fig_handle.set_xlim(1, 9)
#     # fig_handle.set_ylim(0, 400000)
#     # plt.show()
#     # +++++++++++++++++++++++++++++++++++
#     fig_handle = sns.histplot(
#         data=pd.DataFrame({'SecStr code': np.array(X).flatten()}),
#         x='SecStr code', bins=np.array([i-0.5 for i in range(0,8+3,1)])
#         # binwidth=1,
#     )
#     fig = fig_handle.get_figure()
#     fig_handle.set_xlim(0, 9)
#     fig_handle.set_ylim(0, 400000)
#     plt.show()
#     # plt.xlabel('SecStr Code')
#     # outname = prefix+'TrainSet_6_AA_Code_Dist.jpg'
#     # if ControlKeys['SlientRun']==1:
#     #     plt.savefig(outname, dpi=200)
#     # else:  
#     #     plt.show()
#     # #
#     # plt.close()
#     # +++++++++++++++++++++++++++++++++++
    
    
#     print ("#################################")
#     print ("DICTIONARY X")
#     dictt=tokenizer_X.get_config()
#     print (dictt)
#     num_wordsX = len(tokenizer_X.word_index) + 1

#     print ("################## X max token: ",num_wordsX )

#     X=np.array(X)
#     print ("sample X data", X[0])

#     seqs = protein_df.Sequence.values
            
#     #create and fit tokenizer for AA sequences
#     if tokenizer_y==None:
#         tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n' )
#         tokenizer_y.fit_on_texts(seqs)
        
#     y_data = tokenizer_y.texts_to_sequences(seqs)
    
#     y_data= sequence.pad_sequences(y_data,  maxlen=max_length, padding='post', truncating='post')  
    
#     # # --------------------------------------------------------------------
#     # fig_handle =sns.distplot(np.array(y_data).flatten(), bins=21,kde=False, rug=False,norm_hist=False,axlabel='Secondary structure code')
#     # fig = fig_handle.get_figure()
#     # fig_handle.set_xlim(1, 21)
#     # fig_handle.set_ylim(0, 100000)
#     # plt.xticks(range (1,21))
#     # plt.show()
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     fig_handle = sns.histplot(
#         data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
#         x='AA code', bins=np.array([i-0.5 for i in range(0,20+3,1)])
#         # binwidth=1,
#     )
#     fig = fig_handle.get_figure()
#     fig_handle.set_xlim(0, 21)
#     fig_handle.set_ylim(0, 100000)
#     plt.show()
    
    
#     print ("#################################")
#     print ("DICTIONARY y_data")
#     dictt=tokenizer_y.get_config()
#     print (dictt)
#     num_words = len(tokenizer_y.word_index) + 1

#     print ("################## y max token: ",num_words )
#     print ("y data shape: ", y_data.shape)
#     print ("X data shape: ", X.shape)
#     #revere
#     print ("TEST REVERSE: ")

#     y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
#     for iii in range (len(y_data_reversed)):
#         y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
#     print ("Element 0", y_data_reversed[0])

#     print ("Number of y samples",len (y_data_reversed) )
#     print ("Original: ", y_data[:3,:])

#     print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
#     print ("Len 0 as example: ", len (y_data_reversed[0]) )
#     print ("Len 2 as example: ", len (y_data_reversed[2]) )

#     if maxdata<y_data.shape[0]:
#         print ('select subset...', maxdata )
#         X=X[:maxdata]
#         y_data=y_data[:maxdata]
#         print ("new shapes: ", X.shape, y_data.shape)
    
#     X_train, X_test, y_train, y_test = train_test_split(X,  y_data , test_size=split,random_state=235)
    
#     train_dataset = RegressionDataset(torch.from_numpy(X_train).float()/Xnormfac, torch.from_numpy(y_train).float()/ynormfac) #/ynormfac)
    
#     fig_handle = sns.distplot(torch.from_numpy(y_train.flatten()),bins=25,kde=False, 
#                               rug=False,norm_hist=False,axlabel='y labels')
#     fig = fig_handle.get_figure()
#     plt.show()
    
#     fig_handle = sns.distplot(torch.from_numpy(X_train.flatten()),bins=25,kde=False, 
#                               rug=False,norm_hist=False,axlabel='X labels')
#     fig = fig_handle.get_figure()
#     plt.show()
 
#     test_dataset = RegressionDataset(torch.from_numpy(X_test).float()/Xnormfac, torch.from_numpy(y_test).float()/ynormfac)
    
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
#     train_loader_noshuffle = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_)
    
#     return train_loader, train_loader_noshuffle, test_loader, tokenizer_y , tokenizer_X

