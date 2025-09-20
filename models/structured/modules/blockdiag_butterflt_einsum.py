import torch
from einops import rearrange

def low_rank_project(M, rank):
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U_scaled = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt_scaled = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U_scaled, Vt_scaled

def blockdiag_butterfly_project_einsum_rank(M, nblocks1, nblocks2, rank):
    M_batched = _reshape_matrix_to_blocks(M, nblocks1, nblocks2)
    U, Vt = low_rank_project(M_batched, rank=rank)
    w1_bfly = _reshape_vt_to_butterfly(Vt)
    w2_bfly = _reshape_u_to_butterfly(U)
    return w1_bfly, w2_bfly

def _reshape_matrix_to_blocks(M, nblocks1, nblocks2):
    return rearrange(M, '(l j) (k i) -> k j l i', k=nblocks1, j=nblocks2)

def _reshape_vt_to_butterfly(Vt):
    return rearrange(Vt, 'k j r i -> k (r j) i')

def _reshape_u_to_butterfly(U):
    return rearrange(U, 'k j l r -> j l (k r)')