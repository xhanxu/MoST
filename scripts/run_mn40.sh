GPU_ID=2
PATH_CKPT=/datawaha/cggroup/hanx0b/PointGPT/pretrained_ckpt/post_pretrained.pth

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/PointGPT-L/finetune_modelnet_peft.yaml \
--exp_name MoST_PointGPT_mn40_L_code_release_final_v4 \
--ckpts $PATH_CKPT
