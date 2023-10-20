# ==========================================================
# Utility functions
# ==========================================================
import os
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import numpy as np
import math
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

import torch
from einops import rearrange
import esm
# =========================================================
# create a folder path if not exist
def create_path(this_path):
    if not os.path.exists(this_path):
        print('Creating the given path...')
        os.mkdir (this_path)
        path_stat = 1
        print('Done.')
    else:
        print('The given path already exists!')
        path_stat = 2
    return path_stat
            
# ==========================================================

# measure the model size
def params (model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print ("Total parameters: ", pytorch_total_params," trainable parameters: ", pytorch_total_params_trainable)
    
# ==========================================================
# initialization function for dict for models
def prepare_UNet_keys(write_dict):
    # if not setted, using the default
    Full_Keys=['dim', 'text_embed_dim', 'num_resnet_blocks', 'cond_dim', 'num_image_tokens', 'num_time_tokens', 'learned_sinu_pos_emb_dim', 'out_dim', 'dim_mults', 'cond_images_channels', 'channels', 'channels_out', 'attn_dim_head', 'attn_heads', 'ff_mult', 'lowres_cond', 'layer_attns', 'layer_attns_depth', 'layer_attns_add_text_cond', 'attend_at_middle', 'layer_cross_attns', 'use_linear_attn', 'use_linear_cross_attn', 'cond_on_text', 'max_text_len', 'init_dim', 'resnet_groups', 'init_conv_kernel_size', 'init_cross_embed', 'init_cross_embed_kernel_sizes', 'cross_embed_downsample', 'cross_embed_downsample_kernel_sizes', 'attn_pool_text', 'attn_pool_num_latents', 'dropout', 'memory_efficient', 'init_conv_to_final_conv_residual', 'use_global_context_attn', 'scale_skip_connection', 'final_resnet_block', 'final_conv_kernel_size', 'cosine_sim_attn', 'self_cond', 'combine_upsample_fmaps', 'pixel_shuffle_upsample', 'beginning_and_final_conv_present']
    # initialization
    PKeys={}
    for key in Full_Keys:
        PKeys[key]=None
    # modify if keys are provided
    for write_key in write_dict.keys():
        if write_key in PKeys.keys():
            PKeys[write_key]=write_dict[write_key]
        else:
            print("Wrong key found: ", write_key)
        
    return PKeys

def prepare_ModelB_keys(write_dict):
    Full_Keys=['timesteps', 'dim', 'pred_dim', 'loss_type', 'elucidated', 'padding_idx', 'cond_dim', 'text_embed_dim', 'input_tokens', 'sequence_embed', 'embed_dim_position', 'max_text_len', 'cond_images_channels', 'max_length', 'device']
    # initialization
    PKeys={}
    for key in Full_Keys:
        PKeys[key]=None
    # modify if keys are provided
    for write_key in write_dict.keys():
        if write_key in PKeys.keys():
            PKeys[write_key]=write_dict[write_key]
        else:
            print("Wrong key found: ", write_key)
        
    return PKeys

def modify_keys(old_dict,write_dict):
    new_dict = old_dict.copy()
    for w_key in write_dict.keys():
        if w_key in old_dict.keys():
            new_dict[w_key]=write_dict[w_key]
        else:
            print("Alien key found: ", w_key)
    return new_dict

# ==========================================================
# mix two NForce record for a given AA length
# ==========================================================
def mixing_two_FORCE_for_AA_Len(NGap1,Force1,NGap2,Force2,LenAA,mix_fac):
    N = np.amax([len(NGap1), len(NGap2)])
    N_Base = math.ceil(N*2)
    fun_PI_0 = PchipInterpolator(NGap1,Force1)
    fun_PI_1 = PchipInterpolator(NGap2,Force2)
    xx=np.linspace(0,1,N_Base)
    yy=fun_PI_0(xx)*mix_fac+fun_PI_1(xx)*(1-mix_fac)
    fun_PI = PchipInterpolator(xx,yy)
    # discrete result
    x1=np.linspace(0,1,LenAA+1)
    y1=fun_PI(x1)
    return fun_PI, x1, y1

# =========================================================
#
# =========================================================
def get_Model_A_error (fname, cond, plotit=True, ploterror=False):
    
    sec_structure,sec_structure_3state, sequence=get_DSSP_result (fname)
    sscount=[]
    length = len (sec_structure)
    sscount.append (sec_structure.count('H')/length)
    sscount.append (sec_structure.count('E')/length)
    sscount.append (sec_structure.count('T')/length)
    sscount.append (sec_structure.count('~')/length)
    sscount.append (sec_structure.count('B')/length)
    sscount.append (sec_structure.count('G')/length)
    sscount.append (sec_structure.count('I')/length)
    sscount.append (sec_structure.count('S')/length)
    sscount=np.asarray (sscount)
    
    error=np.abs(sscount-cond)
    print ("Abs error per SS structure type (H, E, T, ~, B, G, I S): ", error)

    if ploterror:
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        plt.plot (error, 'o-', label='Error over SS type')
        plt.legend()
        plt.ylabel ('SS content')
        plt.show()

    x=np.linspace (0, 7, 8)
    
    sslabels=['H','E','T','~','B','G','I','S']
    
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    
    ax.bar(x-0.15, cond, width=0.3, color='b', align='center')
    ax.bar(x+0.15, sscount, width=0.3, color='r', align='center')
   
    ax.set_ylim([0, 1])
    
    plt.xticks(range(len(sslabels)), sslabels, size='medium')
    plt.legend (['GT','Prediction'])
    
    plt.ylabel ('SS content')
    plt.show()
    
######################## 3 types

    sscount=[]
    length = len (sec_structure)
    sscount.append (sec_structure_3state.count('H')/length)
    sscount.append (sec_structure_3state.count('E')/length)
    sscount.append (sec_structure_3state.count('~')/length)
    cond_p=[np.sum([cond[0],cond[5], cond[6]]), np.sum ([cond[1], cond[4]]), np.sum([cond[2],cond[3],cond[7]]) ] 
                   
    print ("cond 3type: ",cond_p)
    sscount=np.asarray (sscount)
    
    error3=np.abs(sscount-cond_p)
    print ("Abs error per 3-type SS structure type (C, H, E): ", error)
    
    if ploterror:
        fig, ax = plt.subplots(1, 1, figsize=(6,3))

        plt.plot (error3, 'o-', label='Error over SS type')
        plt.legend()
        plt.ylabel ('SS content')
        plt.show()
    
    
    x=np.linspace (0,2, 3)
    
    sslabels=['H','E', '~' ]
    
    #ax = plt.subplot(111, figsize=(4,4))
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    
                  
    ax.bar(x-0.15, cond_p, width=0.3, color='b', align='center')
    ax.bar(x+0.15, sscount, width=0.3, color='r', align='center')
   
    ax.set_ylim([0, 1])
    
    plt.xticks(range(len(sslabels)), sslabels, size='medium')
    plt.legend (['GT','Prediction'])
    
    plt.ylabel ('SS content')
    plt.show()
    
    return error

def get_DSSP_result (fname):
    pdb_list = [fname]

    # parse structure
    p = PDBParser()
    for i in pdb_list:
        structure = p.get_structure(i, fname)
        # use only the first model
        model = structure[0]
        # calculate DSSP
        dssp = DSSP(model, fname, file_type='PDB' )
        # extract sequence and secondary structure from the DSSP tuple
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]

        #print(i)
        #print(sequence)
        #print(sec_structure)
        #
        # The DSSP codes for secondary structure used here are:
        # =====     ====
        # Code      Structure
        # =====     ====
        # H         Alpha helix (4-12)
        # B         Isolated beta-bridge residue
        # E         Strand
        # G         3-10 helix
        # I         Pi helix
        # T         Turn
        # S         Bend
        # ~         None
        # =====     ====
        #
        
        sec_structure = sec_structure.replace('-', '~')
        sec_structure_3state=sec_structure

        
        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        sec_structure_3state = sec_structure_3state.replace('H', 'H') #0
        sec_structure_3state = sec_structure_3state.replace('E', 'E')
        sec_structure_3state = sec_structure_3state.replace('T', '~')
        sec_structure_3state = sec_structure_3state.replace('~', '~')
        sec_structure_3state = sec_structure_3state.replace('B', 'E')
        sec_structure_3state = sec_structure_3state.replace('G', 'H') #5
        sec_structure_3state = sec_structure_3state.replace('I', 'H') #6
        sec_structure_3state = sec_structure_3state.replace('S', '~')
        return sec_structure,sec_structure_3state, sequence
    
    
def string_diff (seq1, seq2):    
    return   sum(1 for a, b in zip(seq1, seq2) if a != b) + abs(len(seq1) - len(seq2))


# ============================================================
# on esm, rebuild AA sequence from embedding
# ============================================================
import esm

def decode_one_ems_token_rec(this_token, esm_alphabet):
    # print( (this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==100).nonzero(as_tuple=True)[0]==None )

    id_b=(this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0]
    id_e=(this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0]
    
    
    if len(id_e)==0:
        # no ending for this one, so id_e points to the end
        id_e=len(this_token)
    else:
        id_e=id_e[0]
    if len(id_b)==0:
        id_b=0
    else:
        id_b=id_b[-1]

    this_seq = []
    # this_token_used = []
    for ii in range(id_b+1,id_e,1):
        # this_token_used.append(this_token[ii])
        this_seq.append(
            esm_alphabet.get_tok(this_token[ii])
        )
        
    this_seq = "".join(this_seq)

    # print(this_seq)    
    # print(len(this_seq))
    # # print(this_token[id_b+1:id_e]) 
    return this_seq


def decode_many_ems_token_rec(batch_tokens, esm_alphabet):
    rev_y_seq = []
    for jj in range(len(batch_tokens)):
        # do for one seq: this_seq
        this_seq = decode_one_ems_token_rec(
            batch_tokens[jj], esm_alphabet
            )
        rev_y_seq.append(this_seq)
    return rev_y_seq

# ++ for omegafold sequence: treat unknows as X
uncomm_idx_list = [0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# this one decide the beginning and ending AUTOMATICALLY
def decode_one_ems_token_rec_for_folding(
    this_token, 
    this_logits, 
    esm_alphabet, 
    esm_model):
    
    # print( (this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==100).nonzero(as_tuple=True)[0]==None )
    
    # 1. use this_token to find the beginning and ending
    # 2. to logits to generate tokens that ONLY contains foldable AAs
    #
    id_b_0=(this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0]
    id_e_0=(this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0]
    
    # ------------------------------------------------------------------
    # principle: 
    # 1. begin at 0th
    # 2. end as soon as possible: relay on that the first endding is learned
    id_b = 0
    #
    if len(id_e_0)==0:
        id_e=len(this_token)
    else:
        id_e=id_e_0[0]
    # correct if needed
    if id_e<=id_b+1:
        if len(id_e_0)>1:
            id_e=id_e_0[1]
        else:
            id_e=len(this_token)
    # -------------------------------------------------------------------
    
    # # ------------------------------------------------------------------
    # # not perfect
    # # principle: 
    # # 1. begin as late as possible
    # # 2. end as soon as possible
    # #
    # if len(id_b_0)==0:
    #     id_b=0
    # else:
    #     id_b=id_b_0[-1]
    # # so, beginning is set
    # # looking for the nearest ending signal if we can find one
    # # 1. pick those in id_e that id_b<id_e
    # id_e_1=[]
    # for this_e in id_e_0:
    #     if this_e>id_b:
    #         id_e_1.append(this_e)
    # # 2. check what we find
    # if len(id_e_1)==0:
    #     # no endding, id_e points to the end
    #     id_e=len(this_token)
    # else:
    #     # otherwise, find endding point and pick the first one
    #     id_e=id_e_1[0]
    # # 3. if id_b+1==id_e, we still get nothing. So, this is a fake fix
    # if id_e==id_b+1:
    #     if len(id_e_1)>1:
    #         id_e=id_e_1[1]
    #     else:
    #         id_e=len(this_token)
    # # --------------------------------------------------------------------
        
    # if id_b>id_e:
    # for debug:
    print("start at: ", id_b)
    print("end at: ", id_e)
        
    # along the sequence, we pick only index [id_b+1:id_e]. This exclude the <cls> and <eos>
    use_logits = this_logits[id_b+1:id_e] # (seq_len_eff, token_len)
    use_logits[:,uncomm_idx_list]=-float('inf')
    use_token = use_logits.max(1).indices
    
    # print(use_token)
    
    this_seq = []
    # this_token_used = []
    # for ii in range(id_b+1,id_e,1):
    for ii in range(len(use_token)):
        # this_token_used.append(this_token[ii])
        # print(esm_alphabet.get_tok(use_token[ii]))
        # print(ii)
        this_seq.append(
            esm_alphabet.get_tok(use_token[ii])
        )
        
    this_seq = "".join(this_seq)
    
#     # generate a foldable sequece
#     # map all uncommon ones into X/24
#     for idx, one_token in enumerate( this_token_used):
#         find_it=0
#         for this_uncomm in uncomm_idx_list:
#             find_id=find_id+(this_uncomm==one_token)
#         #
#         if find_id>0:
#             this_token_used[idx]=24 # 24 means X
#     # translate token into sequences
#     this_seq_foldable=[]
#     for one_token in this_token_used:
#         this_seq_foldable.append(
#             esm_alphabet.get_tok(one_token)
#         )

#     # print(this_seq)    
#     # print(len(this_seq))
#     # # print(this_token[id_b+1:id_e]) 
    # return this_seq, this_seq_foldable
    return this_seq


def decode_many_ems_token_rec_for_folding(
    batch_tokens,
    batch_logits,
    esm_alphabet,
    esm_model):
    
    rev_y_seq = []
    for jj in range(len(batch_tokens)):
        # do for one seq: this_seq
        this_seq = decode_one_ems_token_rec_for_folding(
            batch_tokens[jj], 
            batch_logits[jj],
            esm_alphabet,
            esm_model,
            )
        rev_y_seq.append(this_seq)
    return rev_y_seq


def convert_into_logits(esm_model, result):
    repre=rearrange(
        result,
        'b l c -> b c l'
    )
    with torch.no_grad():
        logits=esm_model.lm_head(repre)
    
    return logits

# this one return the unmodified tokens and logits
def convert_into_tokens(model, result, pLM_Model_Name):
    if pLM_Model_Name=='esm2_t33_650M_UR50D' \
    or pLM_Model_Name=='esm2_t36_3B_UR50D'   \
    or pLM_Model_Name=='esm2_t30_150M_UR50D' \
    or pLM_Model_Name=='esm2_t12_35M_UR50D' :
        
        repre=rearrange(
            result,
            'b c l -> b l c'
        )
        with torch.no_grad():
            logits=model.lm_head(repre) # (b, l, token_dim)
            
        tokens=logits.max(2).indices # (b,l)
        
    else:
        print("pLM_Model is not defined...")
    return tokens,logits
# ++
def convert_into_tokens_using_prob(prob_result, pLM_Model_Name):
    if pLM_Model_Name=='esm2_t33_650M_UR50D' \
    or pLM_Model_Name=='esm2_t36_3B_UR50D'   \
    or pLM_Model_Name=='esm2_t30_150M_UR50D' \
    or pLM_Model_Name=='esm2_t12_35M_UR50D' :
        
        repre=rearrange(
            prob_result,
            'b c l -> b l c'
        )
        # with torch.no_grad():
        #     logits=model.lm_head(repre) # (b, l, token_dim)
        logits = repre
            
        tokens=logits.max(2).indices # (b,l)
        
    else:
        print("pLM_Model is not defined...")
    return tokens,logits


# 
def read_mask_from_input(
    # consider different type of inputs
    # raw data: x_data (sequences)
    # tokenized: x_data_tokenized
    tokenized_data=None, # X_train_batch, 
    mask_value=None,
    seq_data=None,
    max_seq_length=None,
):
    # # old:
    # mask = X_train_batch!=mask_value
    # new
    if seq_data!=None:
        # use the real sequence length to create mask
        n_seq = len(seq_data)
        mask = torch.zeros(n_seq, max_seq_length)
        for ii in range(n_seq):
            this_len = len(seq_data[ii])
            mask[ii,1:1+this_len]=1
        mask = mask==1
    #
    elif tokenized_data!=None:
        n_seq = len(tokenized_data)
        mask = tokenized_data!=mask_value
        # fix the beginning part: 0+content+00, not 00+content+00
        for ii in range(n_seq):
            # get all nonzero index
            id_1 = (mask[ii]==True).nonzero(as_tuple=True)[0]
            # correction for ForcPath, 
            # pick up 0.0 for zero-force padding at the beginning
            mask[ii,1:id_1[0]]=True
    
    return mask

# ++ read one length
def read_one_len_from_padding_vec(
    in_np_array,
    padding_val=0.0,
):
    mask = in_np_array!=padding_val
    id_list_all_1 = mask.nonzero()[0]
    vec_len = id_list_all_1[-1]+1
    
    return vec_len


# this one decide the beginning and ending using mask
def decode_one_ems_token_rec_for_folding_with_mask(
    this_token, 
    this_logits, 
    esm_alphabet, 
    esm_model,
    this_mask,
):
    # translate all logits into tokens then screen the unmaksed part
    
        
    # along the sequence, we pick only index [id_b+1:id_e]. This exclude the <cls> and <eos>
    use_logits = this_logits # (seq_len_eff, token_len)
    use_logits[:,uncomm_idx_list]=-float('inf')
    use_token = use_logits.max(1).indices
    #
    print(use_token)
    use_token = use_token[this_mask==True]
    # print(use_token)
    
    this_seq = []
    # this_token_used = []
    # for ii in range(id_b+1,id_e,1):
    for ii in range(len(use_token)):
        # this_token_used.append(this_token[ii])
        # print(esm_alphabet.get_tok(use_token[ii]))
        # print(ii)
        this_seq.append(
            esm_alphabet.get_tok(use_token[ii])
        )
        
    this_seq = "".join(this_seq)
    
    return this_seq

def decode_many_ems_token_rec_for_folding_with_mask(
    batch_tokens,
    batch_logits,
    esm_alphabet,
    esm_model,
    mask):
    
    rev_y_seq = []
    for jj in range(len(batch_tokens)):
        # do for one seq: this_seq
        this_seq = decode_one_ems_token_rec_for_folding_with_mask(
            batch_tokens[jj], 
            batch_logits[jj],
            esm_alphabet,
            esm_model,
            mask[jj]
            )
        rev_y_seq.append(this_seq)
    return rev_y_seq

# =====================================================
# create new input condition for ForcPath case
# =====================================================
from scipy import interpolate

def interpolate_and_resample_ForcPath(y0,seq_len1):
    seq_len0=len(y0)-1
    x0=np.arange(0., 1.+1./seq_len0, 1./seq_len0)
    f=interpolate.interp1d(x0,y0)
    #
    x1=np.arange(0., 1.+1./seq_len1, 1./seq_len1)
    y1=f(x1)
    #
    resu = {}
    resu['y1']=y1
    resu['x1']=x1
    resu['x0']=x0
    return resu
#
def mix_two_ForcPath(y0,y1,seq_len2):
    seq_len0=len(y0)-1
    x0=np.arange(0., 1.+1./seq_len0, 1./seq_len0)
    seq_len1=len(y1)-1
    x1=np.arange(0., 1.+1./seq_len1, 1./seq_len1)
    f0=interpolate.interp1d(x0,y0)
    f1=interpolate.interp1d(x1,y1)
    #
    x2=np.arange(0., 1.+1./seq_len2, 1./seq_len2)
    y2=(f0(x2)+f1(x2))/1.
    #
    resu={}
    resu['y2']=y2
    resu['x2']=x2
    resu['x1']=x1
    resu['x0']=x0
    return resu
#
# =====================================================
# load in function for language model
# =====================================================
import esm

def load_in_pLM(pLM_Model_Name,device):
    #
    # ++ for pLM
    if pLM_Model_Name=='trivial':
        pLM_Model=None
        esm_alphabet=None
        len_toks=0
        esm_layer=0

    elif pLM_Model_Name=='esm2_t33_650M_UR50D':
        # dim: 1280
        esm_layer=33
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)

    elif pLM_Model_Name=='esm2_t36_3B_UR50D':
        # dim: 2560
        esm_layer=36
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)

    elif pLM_Model_Name=='esm2_t30_150M_UR50D':
        # dim: 640
        esm_layer=30
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)

    elif pLM_Model_Name=='esm2_t12_35M_UR50D':
        # dim: 480
        esm_layer=12
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)

    else:
        print("pLM model is missing...")
    
    return pLM_Model, esm_alphabet, esm_layer, len_toks