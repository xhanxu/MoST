GPU_ID=0
PATH_CKPT=/datawaha/cggroup/hanx0b/PointGPT/pretrained_ckpt/post_pretrained.pth


CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/PointGPT-L/finetune_scan_hardest_peft.yaml \
--exp_name MoST_PointGPT_hardest_L_code_release_final_v4 \
--ckpts $PATH_CKPT

