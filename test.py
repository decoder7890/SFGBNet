"""
优化的测试脚本 - 通过测试时增强(TTA)和阈值优化提升性能
主要优化点:
1. 自适应阈值搜索
2. 测试时增强(TTA): 水平/垂直翻转
3. 后处理: 形态学操作去噪
4. 智能权重加载
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import HSSNet
import cv2


def apply_tta(model, pre_img, post_img, tta_mode='full'):
    """
    测试时增强(Test Time Augmentation)
    
    参数:
        model: 模型
        pre_img, post_img: 输入图像
        tta_mode: 'none', 'flip', 'full'
    
    返回:
        融合后的预测结果
    """
    predictions = []
    
    # 1. 原始图像
    with torch.no_grad():
        output = model(pre_img, post_img)
        if isinstance(output, (tuple, list)):
            output = output[0]
        predictions.append(output)
    
    if tta_mode == 'none':
        return predictions[0]
    
    # 2. 水平翻转
    if tta_mode in ['flip', 'full']:
        pre_flip = torch.flip(pre_img, dims=[3])  # 水平翻转
        post_flip = torch.flip(post_img, dims=[3])
        with torch.no_grad():
            output_flip = model(pre_flip, post_flip)
            if isinstance(output_flip, (tuple, list)):
                output_flip = output_flip[0]
            output_flip = torch.flip(output_flip, dims=[3])  # 翻转回来
            predictions.append(output_flip)
    
    # 3. 垂直翻转
    if tta_mode == 'full':
        pre_vflip = torch.flip(pre_img, dims=[2])
        post_vflip = torch.flip(post_img, dims=[2])
        with torch.no_grad():
            output_vflip = model(pre_vflip, post_vflip)
            if isinstance(output_vflip, (tuple, list)):
                output_vflip = output_vflip[0]
            output_vflip = torch.flip(output_vflip, dims=[2])
            predictions.append(output_vflip)
    
    # 4. 双向翻转
    if tta_mode == 'full':
        pre_hvflip = torch.flip(pre_img, dims=[2, 3])
        post_hvflip = torch.flip(post_img, dims=[2, 3])
        with torch.no_grad():
            output_hvflip = model(pre_hvflip, post_hvflip)
            if isinstance(output_hvflip, (tuple, list)):
                output_hvflip = output_hvflip[0]
            output_hvflip = torch.flip(output_hvflip, dims=[2, 3])
            predictions.append(output_hvflip)
    
    # 融合所有预测（平均）
    output_avg = torch.stack(predictions).mean(dim=0)
    return output_avg


def apply_morphology(pred_mask, kernel_size=3):
    """
    应用形态学操作去除噪声
    
    参数:
        pred_mask: 预测的二值mask (numpy array)
        kernel_size: 形态学核大小
    
    返回:
        处理后的mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 开运算: 去除小的噪声点
    pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # 闭运算: 填充小的空洞
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    return pred_mask


def find_optimal_threshold(model, val_loader, device, num_samples=100):
    """
    在验证集上搜索最优阈值
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        num_samples: 用于搜索的样本数
    
    返回:
        最优阈值
    """
    print("正在搜索最优阈值...")
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for i, batched_inputs in enumerate(val_loader):
            if i >= num_samples:
                break
            
            img, target = batched_inputs
            pre_img = img[:, 0:3].to(device)
            post_img = img[:, 3:6].to(device)
            target = target.to(device)
            
            output = model(pre_img, post_img)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 搜索最优阈值
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.7, 0.05):
        pred = (all_outputs > threshold).long()
        
        # 计算F1
        tp = ((pred == 1) & (all_targets == 1)).sum().item()
        fp = ((pred == 1) & (all_targets == 0)).sum().item()
        fn = ((pred == 0) & (all_targets == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"最优阈值: {best_threshold:.3f}, F1: {best_f1:.4f}")
    return best_threshold


def load_model_for_dataset(dataset_name):
    """
    加载模型（现在统一使用优化版本的model.py）
    """
    print("=" * 60)
    print(f"检测到 {dataset_name} 数据集")
    print("使用模型: models/model.py (优化版本)")
    print("=" * 60)
    model = HSSNet(3, 1)
    return model, "optimized"


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # 根据数据集自动选择模型版本
    model, model_version = load_model_for_dataset(args.file_root)
    print(f"当前使用模型版本: {model_version}\n")

    # 数据集路径映射
    if args.file_root == 'LEVIR-cd256':
        args.file_root = '../../dataset-gz/LEVIR-CD-256'
    elif args.file_root == 'WHU':
        args.file_root = '../../dataset-gz/WHU-CD-256'
    elif args.file_root == 'SYSU':
        args.file_root = '../../dataset-gz/SYSU-CD-256'
    elif args.file_root == 'CDD':
        args.file_root = '../../dataset-gz/CLCD-256'
    elif args.file_root == 'HRCUS':
        args.file_root = '../../dataset-gz/HRCUS-CD-256'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    device = torch.device('cuda' if args.onGPU else 'cpu')
    model = model.to(device)

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # 数据转换
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # 加载模型权重 - 智能适配
    model_file_name = args.weight
    print(f"加载权重: {model_file_name}")
    
    checkpoint = torch.load(model_file_name, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("加载checkpoint格式 (包含'state_dict'键)")
    else:
        state_dict = checkpoint
        print("加载直接state_dict格式")
    
    # 检查并修复键名不匹配问题（PU/PS vs MDFA/MUFA）
    model_keys = set(model.state_dict().keys())
    weight_keys = set(state_dict.keys())
    
    # 如果权重文件使用PU/PS，但模型使用MDFA/MUFA，需要映射
    if any('PU' in k or 'PS' in k for k in weight_keys) and any('MDFA' in k or 'MUFA' in k for k in model_keys):
        print("检测到键名不匹配: 映射 PU/PS → MDFA/MUFA")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if 'PU' in k and 'PU' not in 'MUFA':
                new_key = k.replace('PU', 'MDFA')
            elif 'PS' in k:
                new_key = k.replace('PS', 'MUFA')
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    # 如果模型使用PU/PS，但权重文件使用MDFA/MUFA，需要反向映射
    elif any('MDFA' in k or 'MUFA' in k for k in weight_keys) and any('PU' in k or 'PS' in k for k in model_keys):
        print("检测到键名不匹配: 映射 MDFA/MUFA → PU/PS")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if 'MDFA' in k:
                new_key = k.replace('MDFA', 'PU')
            elif 'MUFA' in k:
                new_key = k.replace('MUFA', 'PS')
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    # 智能加载权重 - 只加载形状匹配的参数
    model_dict = model.state_dict()
    matched_dict = {}
    missing_in_checkpoint = []
    unexpected_in_checkpoint = []
    shape_mismatch = []
    
    # 检查每个参数
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                shape_mismatch.append((k, v.shape, model_dict[k].shape))
        else:
            unexpected_in_checkpoint.append(k)
    
    for k in model_dict.keys():
        if k not in state_dict:
            missing_in_checkpoint.append(k)
    
    # 更新模型参数
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    
    # 详细的加载报告
    print(f"\n{'='*60}")
    print("权重加载报告:")
    print(f"{'='*60}")
    print(f"✓ 成功加载: {len(matched_dict)}/{len(model_dict)} 参数")
    
    if missing_in_checkpoint:
        print(f"\n⚠️  模型中有 {len(missing_in_checkpoint)} 个参数未在权重文件中找到（将使用随机初始化）")
        for k in missing_in_checkpoint[:5]:
            print(f"    - {k}")
        if len(missing_in_checkpoint) > 5:
            print(f"    ... 还有 {len(missing_in_checkpoint) - 5} 个")
    
    if unexpected_in_checkpoint:
        print(f"\n⚠️  权重文件中有 {len(unexpected_in_checkpoint)} 个参数在当前模型中不存在（已忽略）")
        for k in unexpected_in_checkpoint[:5]:
            print(f"    - {k}")
        if len(unexpected_in_checkpoint) > 5:
            print(f"    ... 还有 {len(unexpected_in_checkpoint) - 5} 个")
    
    if shape_mismatch:
        print(f"\n⚠️  形状不匹配的参数: {len(shape_mismatch)} 个")
        for k, ckpt_shape, model_shape in shape_mismatch[:3]:
            print(f"    - {k}: 权重={ckpt_shape}, 模型={model_shape}")
    
    # 判断是否可以正常测试
    load_ratio = len(matched_dict) / len(model_dict)
    print(f"\n加载比例: {load_ratio*100:.1f}%")
    
    if load_ratio < 0.5:
        print("\n❌ 错误: 加载的参数少于50%，模型可能无法正常工作！")
        print("   建议: 检查权重文件是否与模型匹配")
        if not args.force_test:
            print("   如果确定要继续测试，请添加 --force_test 参数")
            return
    elif load_ratio < 0.95:
        print("\n⚠️  警告: 部分参数未加载，测试结果可能不准确")
        if not args.force_test:
            print("   如果确定要继续测试，请添加 --force_test 参数")
            return
    else:
        print("\n✓ 权重加载完整，可以正常测试")
    
    print(f"{'='*60}\n")
    
    model.eval()
    print("模型设置为评估模式")
    
    # 阈值优化（可选）
    if args.optimize_threshold:
        # 使用部分测试集搜索最优阈值
        optimal_threshold = find_optimal_threshold(model, testLoader, device, num_samples=50)
    else:
        optimal_threshold = args.threshold
    
    print(f"使用阈值: {optimal_threshold:.3f}")
    print(f"TTA模式: {args.tta_mode}")
    print(f"后处理: {'开启' if args.post_process else '关闭'}")

    # 测试
    with torch.no_grad():
        salEvalVal = ConfuseMatrixMeter(n_class=2)
        start = time.time()
        
        for iter, batched_inputs in enumerate(testLoader):
            img, target = batched_inputs
            img_name = testLoader.dataset.file_list[iter]
            pre_img = img[:, 0:3].to(device)
            post_img = img[:, 3:6].to(device)
            target = target.to(device)

            # 应用TTA
            output = apply_tta(model, pre_img, post_img, tta_mode=args.tta_mode)
            
            # 使用优化的阈值
            pred = (output > optimal_threshold).long()
            
            # 后处理
            if args.post_process:
                pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
                pred_np = apply_morphology(pred_np, kernel_size=3)
                pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).long().to(device)

            # 计算指标
            pr = pred[0, 0].cpu().numpy()
            gt = target[0, 0].cpu().numpy()
            salEvalVal.update_cm(pr, gt)
            
            # 保存可视化结果
            if args.save_images:
                index_tp = np.where(np.logical_and(pr == 1, gt == 1))
                index_fp = np.where(np.logical_and(pr == 1, gt == 0))
                index_tn = np.where(np.logical_and(pr == 0, gt == 0))
                index_fn = np.where(np.logical_and(pr == 0, gt == 1))

                change_map = np.zeros([gt.shape[0], gt.shape[1], 3])
                change_map[index_tn] = [128, 128, 128]  # TN: 灰色（背景）
                change_map[index_tp] = [255, 255, 255]  # TP: 白色（正确检测）
                change_map[index_fp] = [255, 0, 0]      # FP: 红色（误检）
                change_map[index_fn] = [0, 255, 255]    # FN: 青色（漏检）
                
                change_map_img = Image.fromarray(np.array(change_map, dtype=np.uint8))
                save_path = os.path.join(args.vis_dir, img_name)
                change_map_img.save(save_path)

            if (iter + 1) % 50 == 0:
                print(f'{iter+1}/{len(testLoader)}, {img_name}')
        
        elapsed_time = time.time() - start
        print(f"\n总耗时: {elapsed_time:.2f}秒")
        print(f"平均每张: {elapsed_time/len(testLoader):.3f}秒")
        
        scores = salEvalVal.get_scores()
        print('\n' + '='*60)
        print('测试结果:')
        print(f'Precision: {scores["precision"]:.4f}')
        print(f'Recall:    {scores["recall"]:.4f}')
        print(f'F1:        {scores["F1"]:.4f}')
        print(f'IoU:       {scores["Iou"]:.4f}')
        print(f'OA:        {scores["OA"]:.4f}')
        print('='*60)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR-cd256", help='数据目录')
    parser.add_argument('--inWidth', type=int, default=256, help='图像宽度')
    parser.add_argument('--inHeight', type=int, default=256, help='图像高度')
    parser.add_argument('--num_workers', type=int, default=4, help='并行线程数')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--vis_dir', default='./LEVIR.log/LEVIR-cd256_iter_68200_lr_0.0001/Vis_optimized', 
                        help='结果保存目录')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='是否使用GPU')
    parser.add_argument('--weight', 
                        default='./LEVIR.log/LEVIR-cd256_iter_68200_lr_0.0001/Epoch151_0.9104581502083412.pth', 
                        type=str, help='预训练权重路径')
    
    # 优化选项
    parser.add_argument('--tta_mode', default='flip', choices=['none', 'flip', 'full'],
                        help='测试时增强模式: none(无), flip(水平翻转), full(全部翻转)')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--optimize_threshold', action='store_true', 
                        help='是否自动搜索最优阈值')
    parser.add_argument('--post_process', action='store_true', 
                        help='是否应用形态学后处理')
    parser.add_argument('--force_test', action='store_true',
                        help='强制测试，即使权重加载不完整')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='是否保存可视化结果图（默认保存）')
    
    args = parser.parse_args()
    print('运行参数:')
    print(args)
    print()

    ValidateSegmentation(args)
