META_INTERNAL=False torchrun --nnodes=1 --nproc_per_node=1 train.py \
--train_dataset " \
400 @ MVDataset(split='train', ROOT='trajectories/scannet_0.3_0.7', aug_crop=16, mask_bg='rand', resolution=(512, 384), transform=ImgNorm, num_views=6, num_render_views=2, random_order = True, random_render_order = False, dps_name = 'dps_train.h5', random_nv_nr = [[6,2]])" \
--test_dataset " \
MVDataset(n_all=5, split='test', ROOT='trajectories/habitatSim_test', resolution=(512,384), seed=777, fix_order=True, num_views=10, num_render_views=6, render_start = 4, dps_name = 'dps_test.h5', tb_name = 'hs_tdf_2_testFull_30_6')" \
--model "AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})" \
--train_criterion "GSRenderLoss(L21, norm_mode='avg_dis', mv = True, scale_scaled = True, use_gt_pcd = False, lpips_coeff = 1.0, rgb_coeff = 1.0, use_img_rgb = True, local_loss_coeff=0.0) + ConfLoss(Regr3D(L21, norm_mode='avg_dis', mv = True), alpha=0.2)" \
--test_criterion "GSRenderLoss(L21, norm_mode='avg_dis', mv = True, render_included = True, scale_scaled = True, use_img_rgb = True, local_loss_coeff=0.0) + Regr3D_ScaleShiftAllInv(L21, gt_scale=True, mv = True)" \
--pretrained "/fs/gamma-projects/3dnvs_gamma/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
--lr 1e-5 --min_lr 0.000001 --warmup_epochs 0 --epochs 20 --batch_size 1 --accum_iter 4 --save_freq 1 --keep_freq 5 --eval_freq 5 --num_workers 16 \
--output_dir outputs/MVD_human_512_dpt_debug
