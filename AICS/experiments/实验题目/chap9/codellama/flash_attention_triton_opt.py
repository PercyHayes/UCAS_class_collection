"""
Fused Attention
===============
 
This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team
 
Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
 
"""
 
import torch
import torch_mlu
 
import triton
import triton.language as tl
#from genesis.Python.Test.Common.utils import reset_tmp_dir
import time
import numpy as np
 
@triton.jit
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        #TODO:指定以下类型为编译时常量
        BLOCK_M:____________________,
        BLOCK_DMODEL: ______________,
        BLOCK_N: ______________,  #
        STAGE: ______________,
        offs_m: ______________,
        offs_n: ______________,  #
        N_CTX: ______________,
        IS_DIVISIBLE: ______________):
    # range of values handled by this stage
    if STAGE == 1:
        #TODO: 处理从0到start_m*BLOCK_M的范围
        lo, hi = ______________________________________________
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        #TODO: # 确保lo是BLOCK_M的整数倍
        lo = ______________________________________________
    # causal = False
    else:
        lo, hi = 0, N_CTX
    #TODO: 将 K_block_ptr 指针向下移动 lo 行，列索引不变，指向当前处理的 K 矩阵的起始行
    K_block_ptr = ______________________________________________
    #TODO: 将 V_block_ptr 指针向右移动 lo 列，行索引不变，指向当前处理的 V 矩阵的起始列
    V_block_ptr = ______________________________________________
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):         # 处理 kv上的 (N_CTX/BLOCK_N) 数据
        #tl.device_print("------------>\n")
        #TODO: 确保 start_n 是 BLOCK_N 的整数倍
        start_n =  ______________________________________________
        # -- compute qk ----
        if IS_DIVISIBLE:
            #TODO:从 K_block_ptr 加载数据到 k
            k = ______________________________________________
        else:
            #TODO:从 K_block_ptr 加载数据到 k
            k = _____________(_____________, boundary_check=(0, 1), padding_option="zero")
        #TODO:创建一个大小为 [BLOCK_N, BLOCK_M] 的全零张量，数据类型为 float32
        qk = ______________________________________________
        #TODO: 计算 q 和 k 的点积并加到 qk上
        qk += ______________________________________________
        if STAGE == 2:#用掩码阻止某些计算
            mask = offs_m[None, :] >= (start_n + offs_n[:, None])
            #TODO: 应用掩码和缩放因子到 qk上
            qk = __________________________tl.where(mask, 0, -1.0e6)
            #TODO: 计算 m_ij，取 m_i 和 qk 中每列的最大值，确保不小于零
            m_ij = ______________________________________________
            #TODO: 从 qk 中减去 m_ij，广播 m_ij 以匹配 qk 的形状，进行归一化处理
            qk -= ______________________________________________
        else:
            #TODO:计算 qk 的每列最大值，并将其乘以缩放因子,并将结果与m_i进行比较 
            m_ij = ______________________________________________
            #TODO:更新qk，保持数值稳定性，防止数值过大
            qk = ______________________________________________
        #TODO:计算 qk 的二次指数（以 2 为底的指数）
        p = __________________________________________
        #TODO: 计算 p 的按行求和结果
        l_ij = ________________________________________
        # -- update m_i and l_i
        #TODO: # 计算 alpha，表示当前 m_i 和 m_ij 之间的指数差异，作为权重调整因子。
        alpha = ________________________________________
        if IS_DIVISIBLE:
            #TODO:从 K_block_ptr 加载数据到 k
            k = ______________________________________________
        else:
            #TODO:从 K_block_ptr 加载数据到 k
            k = _____________(_____________, boundary_check=(0, 1), padding_option="zero")
        #TODO:  将 p 转换为 float16 类型以节省内存和加快计算
        qk_wram = ______________________________________
        #TODO: # 计算加权值 qkv(通过对值向量 v 和转换后的注意力权重 qk_wram 进行点积得到)。
        qkv =  ______________________________________
        # -- update output accumulator --
        #TODO： 更新输出累加器 acc，通过乘以 alpha 来调整之前的累加结果
        acc = ______________________________________
        #TODO： 将当前的 qkv 值加到 acc 中，累加得到最终输出
        acc += ______________________________________
        # update m_i and l_i
        m_i = m_ij
        #TODO: # 更新 l_i，通过加权之前的 l_i 和当前计算的 l_ij
        l_i = ______________________________________
        #TODO: 将 V_block_ptr 向右移动 BLOCK_N 个位置，准备下一次加载。
        V_block_ptr = ______________________________________
        #TODO: 将 K_block_ptr 向下移动 BLOCK_N 个位置，为下一步计算做准备
        K_block_ptr = ______________________________________
 
    return acc, l_i, m_i
 
 
@triton.jit
def _attn_eff_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        Mask_block_ptr,
        #TODO:指定以下类型为编译时常量
        BLOCK_M: _________________,
        BLOCK_DMODEL: _________________,
        BLOCK_N: _________________,  #
        STAGE: _________________,
        offs_m: _________________,
        offs_n: _________________,  #
        N_CTX: _________________,
        IS_DIVISIBLE: _________________,):
        
    # causal = True
    if STAGE == 1:
        #TODO:处理从0到start_m*BLOCK_M的范围
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        #TODO: 保证 lo 是 BLOCK_M 的倍数
        lo = ____________________________________
    # causal = False
    else:
        lo, hi = 0, N_CTX
    #TODO: 将 K_block_ptr 指针向下移动 lo 行，列索引不变，指向当前处理的 K 矩阵的起始行
    K_block_ptr = ______________________________________________
    #TODO: 将 V_block_ptr 指针向右移动 lo 列，行索引不变，指向当前处理的 V 矩阵的起始列
    V_block_ptr = ______________________________________________
    #TODO: 将 Mask_block_ptr指针向下移动 lo 行，以对齐当前处理的掩码块位置
    Mask_block_ptr =______________________________________________
    # loop over k, v and update accumulator
    #TODO:# 在范围 lo 到 hi 之间循环，步长为 BLOCK_N
    __________________________________________________:         # 处理 kv上的 (N_CTX/BLOCK_N) 数据
        #tl.device_print("----- mask ----->\n")
        #TODO: 确保 start_n 是 BLOCK_N 的整数倍
        start_n =  ______________________________________________
        # -- compute qk ----
        #TODO: 加载键向量和掩码块，如果 IS_DIVISIBLE 为真，不进行边界检查
        if IS_DIVISIBLE:
            k = ______________________________________________
            mask =  ______________________________________________
        else:
            k = ________________(________________, boundary_check=(0, 1), padding_option="zero")
            mask = ________________(________________, boundary_check=(0, 1), padding_option="zero")
        #TODO:初始化qk矩阵，创建一个大小为 [BLOCK_N, BLOCK_M] 的全零张量，数据类型为 float32
        qk = ______________________________________________
        #TODO: 计算查询和键的点积
        qk += ______________________________________________
        #tl.device_print("qk0:",qk)
        
        #TODO: 应用掩码和缩放因子到 qk上
        qk = __________________________+ mask*1.44269504
        #TODO: 计算 m_ij，取 m_i 和 qk 中每列的最大值，确保不小于零
        m_ij = ______________________________________________
        #TODO: 从 qk 中减去 m_ij，广播 m_ij 以匹配 qk 的形状，进行归一化处理
        qk -= ______________________________________________
            
        #tl.device_print("qk:",qk)
        #TODO:计算 qk 的二次指数（以 2 为底的指数）
        p = __________________________________________
        #TODO: 计算 p 的按行求和结果
        l_ij = ________________________________________
        #tl.device_print("sum:",l_ij)
        # -- update m_i and l_i
        #TODO: # 计算 alpha，表示当前 m_i 和 m_ij 之间的指数差异，作为权重调整因子。
        alpha = ________________________________________
        if IS_DIVISIBLE:
            #TODO:从 V_block_ptr 加载数据到 v
            v = ______________________________________________
        else:
            #TODO:从 V_block_ptr 加载数据到 v
            v = _____________(_____________, boundary_check=(0, 1), padding_option="zero")
        #qkv = tl.dot(tl.trans(p.to(tl.float16)), v)
        #TODO:  将 p 转换为 float16 类型以节省内存和加快计算
        qk_wram = ______________________________________
        #TODO: # 计算加权值 qkv(通过对值向量 v 和转换后的注意力权重 qk_wram 进行点积得到)。
        qkv =  ______________________________________
        # -- update output accumulator --
        #TODO： 更新输出累加器 acc，通过乘以 alpha 来调整之前的累加结果
        acc = ______________________________________
        #TODO： 将当前的 qkv 值加到 acc 中，累加得到最终输出
        acc += ______________________________________
        # update m_i and l_i
        m_i = m_ij
        #TODO: # 更新 l_i，通过加权之前的 l_i 和当前计算的 l_ij
        l_i = ______________________________________
        #TODO: 将 V_block_ptr 向右移动 BLOCK_N 个位置，准备下一次加载。
        V_block_ptr = ______________________________________
        #TODO: 将 K_block_ptr 向下移动 BLOCK_N 个位置，为下一步计算做准备
        K_block_ptr = ______________________________________
        #TODO: 将 Mask_block_ptr 向下移动 BLOCK_N 行，以准备处理下一个块的掩码数据
        Mask_block_ptr = tl.advance(Mask_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i 
 
@triton.jit
def _attn_eff_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        stride_mm,
        stride_mn,  #
        Z,
        H,  #
        causal_mask,
        #TODO：指定以下类型为编译时常量
        N_CTX: _____________________,  #
        Q_N_CTX: _____________________,
        BLOCK_M: _____________________,  #
        BLOCK_N: _____________________,  #
        BLOCK_DMODEL: _____________________,  #
        STAGE: _____________________,  #
        IS_DIVISIBLE: _____________________):
    
    #TODO: 获取当前核心的唯一标识符，用于区分不同的计算核心
    core_id = _____________________________________
    #TODO: 获取在第一个维度（核心维度）上的核心总数，用于并行计算
    core_dim = _____________________________________
    #TODO: 获取当前集群的唯一标识符，用于区分不同的计算集群
    cluster_id = _____________________________________
    #TODO: 获取在第二个维度（集群维度）上的核心总数，以便进行任务分配和调度
    cluster_dim = _____________________________________

    #TODO: 计算每个上下文的数量
    context_num = _____________________________________
    #TODO: 计算总的注意力头数量
    total_heads = _____________________________________
    #TODO: 每个集群分配的头数量
    task_heads = _____________________________________
    #TODO: 计算剩余的头数（总头数减去每个集群处理的头数的乘积）
    task_remain_heads = _____________________________________
    #TODO: 保证每个集群处理的任务数量至少为1
    task_heads +=__________________________________
    #TODO: 计算当前集群开始处理的头的索引
    task_head_begin = __________________________________
    if cluster_id >= task_remain_heads:
        #TODO: 减少当前集群的任务数量
        task_heads -=  __________________________________
        #TODO:  更新当前集群开始处理的头的索引，以便正确定位到可处理的头
        task_head_begin =__________________________________
    if task_heads <= 0:
        return
 
    #TODO: 计算每个核心需要处理的头的数量
    core_heads = __________________________________
    #TODO: 计算剩余的头数
    core_remain_heads = __________________________________
    #TODO: 保证每个核心处理的任务数量至少为1
    core_heads += __________________________________
    #TODO: 计算当前核心开始处理的头的索引
    core_head_begin = __________________________________
    if core_id >= core_remain_heads:
        #TODO: 减少当前核心的任务数量
        core_heads -= __________________________________
        #TODO: 更新当前核心开始处理的头的索引，以便正确定位到可处理的头
        core_head_begin = __________________________________
    if core_heads <= 0:
        return
    #TODO: 计算实际处理的头的起始索引
    head_begin = __________________________________
    #TODO: 计算实际处理的头的结束索引
    head_end = __________________________________
 
    for head_idx in range(head_begin, head_end):  # 一个core处理 q上的 (Q_N_CTX/BLOCK_M) 数据
        #TODO: 计算当前头的起始索引在上下文中的位置
        start_m = __________________________________
        #TODO: 计算当前头在上下文中的偏移量
        off_hz = __________________________________
        #TODO: 计算当前头的 z 维度偏移
        off_z = __________________________________
        #TODO: 计算当前头的 h 维度偏移
        off_h = __________________________________
        #TODO: 将 off_z 和 off_h 转换为 int64 类型，并分别乘以查询张量的步幅 stride_qz 和 stride_qh，以得到查询张量在内存中的实际位置
        _____________________________________________________________________________
        #TODO: 将 off_z 和 off_h 转换为 int64 类型，并分别乘以键值张量的步幅 stride_kz 和 stride_kh，以得到键值张量在内存中的实际位置
        _____________________________________________________________________________  
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),          
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        #TODO: 仿照以上Q_block_ptr的创建方法，创建K_block_ptr，指定基地址和形状。
        K_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________
        
        #TODO: 仿照以上Q_block_ptr的创建方法，创建V_block_ptr，指定基地址和形状。
        V_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________
     
        #TODO: 仿照以上Q_block_ptr的创建方法，创建Mask_block_ptr，指定基地址和形状。
        Mask_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________

        #TODO: 仿照以上Q_block_ptr的创建方法，创建O_block_ptr，指定基地址和形状。
        O_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________

        # initialize offsets
        #TODO: 计算当前块的行偏移量，offs_m 为从 start_m 开始的连续 BLOCK_M 行的索引
        offs_m = ____________________________
        #TODO: 创建 offs_n，表示当前块的所有列索引，从 0 到 BLOCK_N - 1。
        offs_n = ___________________________
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/ln(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = ______________________________________________
        else:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = _____________(_____________, boundary_check=(0, 1), padding_option="zero")
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            #TODO:调用_attn_eff_fwd_inner函数处理非对角线的部分计算
            acc, l_i, m_i = ____________________________________
        # stage 2: on-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 2 as its STAGE
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            #TODO: 同步线程
            ____________________________
            #TODO:调用__attn_eff_fwd_inner函数处理对角线的部分计算
            acc, l_i, m_i = _________________________________________
        # epilogue
        #TODO:计算 log2(l_i) 并加到 m_i
        m_i += ____________________________________
        #TODO: 计算 l_i 的倒数
        l_i_recip = ____________________________________
        #TODO: 将 acc 矩阵的每个元素乘以 l_i_recip 的每一列
        acc = ____________________________________
        #TODO: 对 acc 矩阵进行转置操作，以便适应后续存储或计算
        acc = ____________________________________
        #TODO: 计算 m_ptrs，作为 M 矩阵中的指针，结合 off_hz 和 offs_m。
        m_ptrs = ____________________________________
        if IS_DIVISIBLE:
            #TODO: 将当前的m_i值存储到m_ptrs指向的位置
            ____________________________________
            #TODO: # 将累加结果acc转换为输出类型并存储到O_block_ptr指向的位置
            _______________( _______________, acc.to(Out.type.element_ty))
        else:
            #TODO: 仅在offs_m小于N_CTX的情况下，将m_i存储到m_ptrs，应用掩码
            ____________________________________
            #TODO:  # 将累加结果acc转换为输出类型并存储到O_block_ptr，进行边界检查
            ____________________________________
 
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,  #
        #TODO：指定以下类型为编译时常量
        N_CTX: _______________________________,  #
        Q_N_CTX: _______________________________,
        BLOCK_M: _______________________________,  #
        BLOCK_N: _______________________________,  #
        BLOCK_DMODEL: _______________________________,  #
        STAGE: _______________________________,  #
        IS_DIVISIBLE: _______________________________):
    #TODO: 获取当前核心的唯一标识符，用于区分不同的计算核心
    core_id = _____________________________________
    #TODO: 获取在第一个维度（核心维度）上的核心总数，用于并行计算
    core_dim = _____________________________________
    #TODO: 获取当前集群的唯一标识符，用于区分不同的计算集群
    cluster_id = _____________________________________
    #TODO: 获取在第二个维度（集群维度）上的核心总数，以便进行任务分配和调度
    cluster_dim = _____________________________________
 
    #TODO:计算Q_N_CTX与BLOCK_M的整除结果，得到上下文的块数量
    context_num = _____________________________________ # 向上取整
    #TODO: 计算总的注意力头数量
    total_heads = _____________________________________
    #TODO: 每个集群分配的头数量
    task_heads = _____________________________________
    #TODO: 计算剩余的头数（总头数减去每个集群处理的头数的乘积）
    task_remain_heads = _____________________________________
    #TODO: 保证每个集群处理的任务数量至少为1
    task_heads +=__________________________________
    #TODO: 计算当前集群开始处理的头的索引
    task_head_begin = __________________________________
    if cluster_id >= task_remain_heads:
        #TODO: 减少当前集群的任务数量
        task_heads -=  __________________________________
        #TODO:  更新当前集群开始处理的头的索引，以便正确定位到可处理的头
        task_head_begin =__________________________________
    else:
        pass
    if task_heads <= 0:
        return
 
    #TODO: 计算每个核心需要处理的头的数量
    core_heads = __________________________________
    #TODO: 计算剩余的头数
    core_remain_heads = __________________________________
    #TODO: 保证每个核心处理的任务数量至少为1
    core_heads += __________________________________
    #TODO: 计算当前核心开始处理的头的索引
    core_head_begin = __________________________________
    if core_id >= core_remain_heads:
        #TODO: 减少当前核心的任务数量
        core_heads -= __________________________________
        #TODO: 更新当前核心开始处理的头的索引，以便正确定位到可处理的头
        core_head_begin = __________________________________
    if core_heads <= 0:
        return
    #TODO: 计算实际处理的头的起始索引
    head_begin = __________________________________
    #TODO: 计算实际处理的头的结束索引
    head_end = __________________________________
 
    for head_idx in range(head_begin, head_end):
        #TODO: 计算当前头的起始索引在上下文中的位置
        start_m = __________________________________
        #TODO: 计算当前头在上下文中的偏移量
        off_hz = __________________________________

        #TODO: 计算当前头的 z 维度偏移
        off_z = __________________________________
        #TODO: 计算当前头的 h 维度偏移
        off_h = __________________________________
        #TODO: 计算查询（Q）的内存偏移量，基于z和h维度的偏移量及其步幅
        q_offset = _______________________________________________________________
        #TODO: 计算KV的内存偏移量，基于z和h维度的偏移量及其步幅
        kv_offset = _______________________________________________________________
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),          
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建K_block_ptr
        K_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________

        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建V_block_ptr
        V_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________
        
        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建O_block_ptr
        O_block_ptr = ____________________________
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________,
        ____________________________

        # initialize offsets
        #TODO: 计算当前块的行偏移量，offs_m 为从 start_m 开始的连续 BLOCK_M 行的索引
        offs_m = ____________________________
        #TODO: 创建 offs_n，表示当前块的所有列索引，从 0 到 BLOCK_N - 1。
        offs_n = ___________________________
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = ______________________________________________
        else:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = _____________(_____________, boundary_check=(0, 1), padding_option="zero")
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            #TODO:调用_attn_fwd_inner函数处理非对角线的部分计算
            acc, l_i, m_i = ____________________________________
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            #TODO: 同步线程
            ____________________________
            #TODO:调用_attn_fwd_inner函数处理对角线的部分计算
            acc, l_i, m_i = _________________________________________
        # epilogue
        #TODO:计算 log2(l_i) 并加到 m_i
        m_i += ____________________________________
        #TODO: 计算 l_i 的倒数
        l_i_recip = ____________________________________
        #TODO: 将 acc 矩阵的每个元素乘以 l_i_recip 的每一列
        acc = ____________________________________
        #TODO: 对 acc 矩阵进行转置操作，以便适应后续存储或计算
        acc = ____________________________________
        #TODO: 计算 m_ptrs，作为 M 矩阵中的指针，结合 off_hz 和 offs_m。
        m_ptrs = ____________________________________
        if IS_DIVISIBLE:
            #TODO: 将当前的m_i值存储到m_ptrs指向的位置
            ____________________________________
            #TODO: # 将累加结果acc转换为输出类型并存储到O_block_ptr指向的位置
            ____________________________________
        else:
            #TODO: 仅在offs_m小于N_CTX的情况下，将m_i存储到m_ptrs，应用掩码
            ____________________________________
            #TODO:  # 将累加结果acc转换为输出类型并存储到O_block_ptr，进行边界检查
            ____________________________________

@triton.jit
def _attn_bwd_preprocess(
        O,
        DO,  #
        Delta,  #
        Z,
        H,
        N_CTX,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_M: ___________________________________,
        D_HEAD: ____________________________________  #
):
    #TODO: 计算当前程序实例在 M 维度上的偏移量
    off_m = ___________________________________
    #TODO: 获取当前程序的第二个维度的 ID，用于头的分配
    off_hz = ___________________________________
    #TODO:  计算每个注意力头的维度上的偏移量off_n，即从 0 到 D_HEAD-1 的范围
    off_n =  ___________________________________
    # load
    o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD +
                off_n[None, :])
    #TODO:# 同样计算偏移量并将 DO 中的数据加载到do,并将数据类型转换为 float32
    do = ___________________________________
    #TODO: 计算 delta，它是 o 和 do的逐元素乘积的和
    delta = ___________________________________
    # write-back
    #TODO: 计算得到的 delta 存储到 Delta 张量中的特定位置
    ___________________________________
 
 
# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,
        #TODO： 指定以下类型为编译时常量
        BLOCK_M1:____________________,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_N1:____________________,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_DMODEL: _______________,  #
        start_n,
        start_m,
        num_steps,  #
        #TODO： 指定以下类型为编译时常量
        MASK: ___________________):
    #TODO: 计算 offs_m，表示当前块内 m 维度上的偏移量
    offs_m = ___________________________________
    #TODO: 计算 offs_n，表示当前块内 n 维度上的偏移量
    offs_n = ___________________________________
    #TODO: 计算 offs_k，用于在模型维度上进行计算和索引
    offs_k = ___________________________________
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    #TODO: 计算 DO 的指针位置
    do_ptrs = _____________________________________________________________
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    #TODO:  静态断言：BLOCK_N1 必须是 BLOCK_M1 的倍数
    __________________________________
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        #TODO:加载qT_ptrs
        qT = ____________________________________________
        # Load m before computing qk to reduce pipeline stall.
        #TODO： 计算当前块 m 维度上的偏移量
        offs_m = ___________________________________
        #  # 从内存中加载 M 中偏移 offs_m 处的数据，用于后续计算
        m = ___________________________________
        #TODO: 计算 k 和 qT 的点积
        qkT = ______________________________
        #TODO:应用指数函数，将 qkT 和 m 的差值转换为以2为底的指数形式
        pT = __________________________________
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            #TODO: 应用掩码，将未来的时间步的 pT 设置为 0
            pT = __________________________________
        #TODO:加载o_ptrs
        do = _________________________
        # Compute dV.
        ppT = pT
        #TODO: 将 ppT 转换为 float16 类型
        ppT = _________________________
        #TODO: 计算 dV，dV 是 ppT 和 do 的点积，将结果累加到 dv 中
        dv += __________________________________
        # D (= delta) is pre-divided by ds_scale.
        #TODO: 从内存中加载 D中偏移 offs_m 处的数据
        Di = ______________________________
        # Compute dP and dS.
        #TODO: 计算值矩阵 v 和 do 的转置的点积，得到对 v 的梯度，结果转换为 float32 类型
        dpT = ______________________________
        #TODO: 计算dsT，通过 pT 乘以 dpT 和Di之间的差值
        dsT = ______________________________
        #TODO: 将 dsT 转换为 float16 类型
        dsT =  ______________________________
        #TODO: 计算 dK，将 dsT 和 qT 的转置的点积结果累加到 dk 中
        dk += ______________________________
        # Increment pointers.
        #TODO: 更新当前块的 m 维度上的起始位置
        curr_m += ______________________________
        #TODO: 更新 qT_ptrs，指向下一个块
        qT_ptrs += ______________________________
        # 更新 do_ptrs，指向下一个块
        do_ptrs += ______________________________
    return dk, dv
 
 
# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_M2: ______________________________,  #
        BLOCK_N2: ______________________________,  #
        BLOCK_DMODEL:______________________________,
        # Filled in by the wrapper.
        start_m,
        start_n,
        num_steps,  #
        #TODO： 指定以下类型为编译时常量
        MASK: ______________________________):
    #TODO: 计算当前块在 m 方向的偏移量
    offs_m = ______________________________
    #TODO: 计算当前块在 n 方向的偏移量
    offs_n = ______________________________
    #TODO: 计算 offs_k，用于在模型维度上进行计算和索引
    offs_k = ______________________________
    #TODO: 计算键矩阵 K 的指针位置
    kT_ptrs = _____________________________________________________________
    #TODO: 计算键矩阵 V 的指针位置
    vT_ptrs =  _____________________________________________________________
    # D (= delta) is pre-divided by ds_scale.
    #TODO:从内存中加载数据 D 在由 offs_m 确定的偏移位置上的值，
    Di =____________________________________________________
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    #TODO: 静态断言：BLOCK_N1 必须是 BLOCK_M1 的倍数
    ____________________________________________________
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        #TODO: 加载kT_ptrs
        kT = _____________________
        #TODO: 加载vT_ptrs
        vT = _____________________
        #TODO: 计算 q 和 kT 的点积
        qk = _____________________
        #TODO:将 qk 和 m 的差值转换为以2为底的指数形式
        p = _____________________
        # Autoregressive masking.
        if MASK:
            #TODO: 计算当前块的 n 方向的偏移量
            offs_n = ________________________________________
            #TODO: 生成掩码，掩盖不需要的元素
            mask = ________________________________________
            #TODO: 应用掩码，将掩盖的部分置为 0
            p = ________________________________________
        # Compute dP and dS.
        #TODO: 计算值矩阵 do 和 vT 的转置的点积，将结果转换为 float32 类型
        dp = ________________________________________
        #TODO: 计算ds，通过 p乘以 dp 和Di之间的差值
        ds = ________________________________________
        #TODO: 将 ds 转换为 float16 类型
        ds = ________________________________________
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        #TODO: 计算 dq，将 ds 和 kT 的转置的点积结果累加到 dq 中
        dq += ________________________________________
        # Increment pointers.
        #TODO: 更新指针，移动到下一个块的位置
        curr_n +=  ___________________________
        kT_ptrs +=  ___________________________
        vT_ptrs +=  ___________________________
    return dq   
 
 
@triton.jit
def _attn_bwd(
        Q,
        K,
        V,
        sm_scale,  #
        DO,  #
        DQ,
        DK,
        DV,  #
        M,
        D,
        # shared by Q/K/V/DO.
        stride_z,
        stride_h,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
    #TODO： 指定以下类型为编译时常量
        BLOCK_M1: ______________________,  #
        BLOCK_N1: ______________________,  #
        BLOCK_M2: ______________________,  #
        BLOCK_N2: ______________________,  #
        BLK_SLICE_FACTOR: ______________________,  #
        BLOCK_DMODEL: ______________________):
    LN2: ______________________ = 0.6931471824645996  # = ln(2)
 
    #TODO: 获取当前程序的批次/头部 ID
    bhid =  ______________________
    #TODO: 计算当前批次或注意力头部 bhid 的起始位置在数据数组 Q, K, V 和其他相关数组中的偏移量,并将其转换为 64 位整数
    off_chz = ______________________
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    #TODO: 获取当前程序的 ID
    pid = ______________________
 
    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
 
    # load scales
    #TODO: 生成一个包含从 0 到 BLOCK_DMODEL - 1 的整数序列，用于表示当前计算块中所有的维度索引
    offs_k =  ___________________________
 
    #TODO: 计算当前处理的起始位置
    start_n = ___________________________
    start_m = start_n
 
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    #TODO: 从 start_n 开始，通过增加从 0 到 BLOCK_N1 的范围值来计算每个块的起始位置，用于确定在块内的列偏移
    offs_n =  ____________________________
    
    #TODO: 初始化一个全零张量 dv，其形状为 [BLOCK_N1, BLOCK_DMODEL]，数据类型为 float32
    dv = ____________________________________________________
    #TODO: 初始化一个全零张量 dk，其形状为 [BLOCK_N1, BLOCK_DMODEL]，数据类型为 float32
    dk = ____________________________________________________
 
    # load K and V: they stay in SRAM throughout the inner loop.
    #TODO: 根据偏移量 offs_n 和 offs_k 以及步长 stride_tok 和 stride_d，从内存中加载键向量 K，并将其存储到变量 k 中
    k = ____________________________________________________
    #TODO: 根据偏移量 offs_n 和 offs_k 以及步长 stride_tok 和 stride_d，从内存中加载值向量 V，并将其存储到变量 v 中
    v = ____________________________________________________
 
    #TODO: 通过将总数据块大小除以每次迭代处理的数据块大小来计算处理所有数据块所需的迭代次数
    num_steps = ____________________________________________________ 
 
    #TODO: 调用函数计算 dk 和 dv（采用mask)
    dk, dv = _______________________________________________________
 
    #TODO： 更新 start_m 
    start_m += __________________________________
    #TODO: 更新num_steps
    num_steps = __________________________________
 
    # Compute dK and dV for non-masked blocks.
    #TODO: 调用函数计算 dk 和 dv（不用采用mask)
    dk, dv = ______________________________________________________
 
    ## 计算 dv 的存储位置指针
    dv_ptrs = ______________________________________________________
    # 将 dv 张量的内容存储到 dv_ptrs 所指向的内存位置
    ______________________________________________________
    
    #TODO:按比例缩放 dk 张量，将其乘以缩放因子
    dk *= ______________________________________________________
    #TODO:计算 dk_ptrs，将其指向 DK 张量在内存中的对应位置
    dk_ptrs = ______________________________________________________
    #TODO: 将缩放后的 dk 张量存储到 dk_ptrs 所指向的内存位置
    ______________________________________________________
 
    #TODO: 更新 start_m(根据pid和BLOCK_M2 确定) 
    start_m = _____________________________________________
    #TODO: 计算结束的列索引
    end_n = _______________________________________________
 
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    #TODO: 从 start_n 开始，通过增加从 0 到 BLOCK_M2 的范围值来计算每个块的起始位置，用于确定在块内的列偏移
    offs_m = _______________________________________________

    #TODO: 从张量 Q 中加载数据，利用 offs_m 和 offs_k 计算内存地址偏移量
    q = _______________________________________________
    #TODO: 初始化 dq 为全零张量，大小为 [BLOCK_M2, BLOCK_DMODEL]，数据类型为 float32
    dq = _______________________________________________
    #TODO: 从张量 DO 中加载数据，利用 offs_m 和 offs_k 计算内存地址偏移量
    do = _______________________________________________
 
    #TODO: 从张量 M 中加载数据，根据 offs_m 计算出要加载的内存地址
    m = _______________________________________________
    m = m[:, None]
 
    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    #TODO:  通过将总数据块大小除以每次迭代处理的数据块大小来计算处理所有数据块所需的迭代次数
    num_steps = __________________________________________________
    #TODO: 调用 _attn_bwd_dq 函数，计算带掩码的 dQ 值
    dq = __________________________________________________
    #TODO: 更新 end_n，减少已经计算过的列数。
    end_n -= __________________________________________________
    # stage 2
    #TODO: 计算无掩码块的步数
    num_steps = __________________________________________________
    #TODO: 再次调用 _attn_bwd_dq 函数，计算没有掩码的 dQ 值
    dq = __________________________________________________
    # Write back dQ.
    #TODO：计算 DQ 的存储位置指针
    dq_ptrs =  __________________________________________________
    #TODO: 利用LN2对计算的 dQ 进行缩放
    dq *=  __________________________________________________
    #TODO: 将缩放后的 dq存储到指定的位置dq_ptrs中。
    __________________________________________________
 
 
empty = torch.empty(128, device="mlu")
 
 
class _attention(torch.autograd.Function):
 
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale,causal_mask):
        # shape constraints
        Lq, Lk, Lv = __________________________________________________
        #TODO: 确保Lq、Lk和Lv张量的长度相同
        __________________________________________________
        #assert Lk in {16, 32, 64, 128}
        #TODO: 创建一个与查询张量q形状相同的空张量
        o =__________________________________________________
   
        # 如果Nram不够，需要改为原始
        # BLOCK_M = 128
        # BLOCK_N = 64 if Lk <= 64 else 32
        
        #TODO: 获取查询张量 q 的上下文长度（倒数第二维的大小）
        q_ctx_len = __________________________________
        #TODO: 获取键张量 k 的上下文长度（倒数第二维的大小）
        kv_ctx_len= __________________________________

        if q_ctx_len <= 128:
            if kv_ctx_len%128==0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 128
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len < 256:
            if kv_ctx_len%64==0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 64
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len >= 256:
            if kv_ctx_len%64==0:
                BLOCK_M = 64
                BLOCK_N = 64
            elif kv_ctx_len%32==0:
                BLOCK_M = 32
                BLOCK_N = 32
            else:
                BLOCK_M = 64
                BLOCK_N = kv_ctx_len
        
        


        #print("------------BLOCK_M:",BLOCK_M)
        #print("------------BLOCK_N:",BLOCK_N)


        num_stages = 4 if Lk <= 64 else 3
        num_warps = 1
        stage = 3 if causal else 1
        if torch.mlu.get_device_capability()[0] == 9:
            num_warps = 8
            # num_stages = 7 if Lk >= 64 else 3
            num_stages = 0
        num_stages = 0
        #grid is coredim clusterdim 1
        grid = (4, 8, 1)
        #TODO: 创建一个与 q 张量形状匹配的空张量 M，同时将 M 张量分配到与 q 张量相同的设备上,并将 M 张量的数据类型设置为 float32。
        M = __________________________________________________

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        #TODO: 获取张量 q 在第三维的大小，并将其赋值给 N_CTX，表示上下文数量。
        N_CTX = ______________________________________________
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True


        if(causal_mask is not None):
            _attn_eff_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                causal_mask.stride(2), 
                causal_mask.stride(3),
                q.shape[0],
                q.shape[1],  #
                causal_mask,
                N_CTX=k.shape[2],  #
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,  #
                BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=Lk,  # D_HEAD
                STAGE=stage,  #
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                num_warps=num_warps,  #
                num_stages=num_stages  #
            )
        else:
            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=k.shape[2],  #
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,  #
                BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=Lk,  # D_HEAD
                STAGE=stage,  #
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                num_warps=num_warps,  #
                num_stages=num_stages  #
            )
 
 
 
        #TODO:保存前向传播中需要反向传播使用的张量q、k、v、o、M以供反向传播时使用
        _________________________________________________
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o
 
    @staticmethod
    def backward(ctx, do):
        #TODO:  从 ctx 对象中恢复在前向传播中保存的张量q, k, v, o, M
        q, k, v, o, M = _________________________________________________
        #TODO: 确保梯度张量do是连续的（即在内存中存储为连续块）
        _________________________________________________
        #TODO: 检查q、k、v、o和do张量的内存步长是否一致，以确保它们的内存布局相同。
        _________________________________________________
        #TODO: 创建与q张量相同形状的新张量dq
        dq =_________________________________________________
        #TODO: 创建与k张量相同形状的新张量dk
        dk =_________________________________________________
        #TODO: 创建与v张量相同形状的新张量dv
        dv =_________________________________________________
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 0
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        #TODO:将arg_k 乘以标量 ctx.sm_scale * RCP_LN进行缩放，更新 arg_k 的值。
        arg_k =_________________________________________________
        PRE_BLOCK = 128
        #assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        #TODO: 创建一个与M张量形状相同的新张量delta
        delta = _________________________________________________
        #TODO: 调用_attn_bwd_preprocess函数，在pre_grid网格上执行反向传播预处理操作
        _________________________________________________
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
 
        return dq, dk, dv, None, None
 
 
attention = _attention.apply
 
 

def test_op():
    torch.manual_seed(20)
    Z, H, N_CTX, D_HEAD=1,128,257,16
    causal=False # 
    causal_mask=[]
    dtype=torch.float16
    use_data_from_file=False
    Q_N_CTX=N_CTX
    
    for n in range(N_CTX,N_CTX+1):
        if (use_data_from_file==False):
            #TODO: 生成随机的q张量，并设置为需要梯度
            q = __________________________________________________________________
            #TODO: 生成随机的k张量，并设置为需要梯度
            k = __________________________________________________________________
            v = (torch.empty((Z, H, n, D_HEAD), dtype=dtype,
                         device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
            if(causal_mask is not None):
                causal_mask = (torch.empty((Z, 1, Q_N_CTX, n), dtype=dtype,
                             device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
                # causal_mask = (torch.zeros((Z, 1, Q_N_CTX, n), dtype=dtype,
                             # device="mlu").requires_grad_()).contiguous()
        else:
            q_np = np.fromfile("query_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            k_np = np.fromfile("key_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            v_np = np.fromfile("value_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)

            #TODO: 将q_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            q =__________________________________________________________________
            #TODO: 将k_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            k = __________________________________________________________________
            #TODO: 将v_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            v = __________________________________________________________________

        sm_scale = 0.5
        #TODO: 生成与q张量相同形状的随机张量
        dout = ____________________________________
        
        print("q:",q.shape)
        print("k:",k.shape)
        print("v:",v.shape)
        print("causal:",causal)
        
        # triton的实现
        st=time.time()    
        #TODO: 调用自定义的高效注意力机制函数，并将结果转换为半精度浮点数
        tri_out = __________________________________________________________________
        ed=time.time()
        print("triton attention cost:",ed-st)
        ##print("tri_out:",tri_out)
        #TODO: 将tri_out展平
        tri_out=____________________________________
        #TODO: 标识张量tri_out中的每个元素是否为 NaN (Not a Number)
        nan_mask = ____________________________________
        #TODO:  检查nan_mask中是否有任何元素为 True
        has_nan = ____________________________________
        #print("tri_out has_nan",has_nan)
        
        
        # sdpa的实现
        st=time.time()
        if(causal_mask is not None): causal=False
        #TODO: 调用 PyTorch 的 scaled_dot_product_attention 函数，计算缩放点积注意力。
        sdpa_output = ___________________________________________________________________________________________
        ed=time.time()
        print("scaled_dot_product_attention attention cost:",ed-st)
        ##print("sdpa_output:",sdpa_output)
        #TODO: 将sdpa_output展平
        sdpa_output=____________________________________
        
        pytorch_valible=True
        if(pytorch_valible==True):
            ## pytorch的实现
            st=time.time() 
            #TODO: 创建一个下三角矩阵 M，大小为 (Q_N_CTX, N_CTX)，其元素为 1
            M = ____________________________________________________
            #TODO: 计算查询张量 q 和键张量 k 的转置的点积
            qk = ____________________________________________________
            #TODO: 将 qk 乘以缩放因子 sm_scale
            p=____________________________________
            if(causal_mask is not None):
                p=p+causal_mask
            elif causal:
                p[:, :, M == 0] = float("-inf")
            
            if(1):
                #TODO: 对 p 应用 softmax，并将结果转换为半精度浮点数
                p = ____________________________________
                #TODO:  计算 p 和值张量 v 的点积，得到 pyt_out
                pyt_out = ____________________________________
            #TODO:将pyt_out进行展平
            pyt_out=____________________________________
            ed=time.time()
            print("pytorch attention cost:",ed-st)
            
            # compare
            #TODO：计算 pyt_out 和 tri_out 之间的绝对误差总和
            abs_tp_error = ____________________________________
            #TODO: 计算 pyt_out 和 tri_out 之间的相对误差，总绝对误差除以两者绝对值和的最小值，避免除以零
            rel_tp_error = ____________________________________
            print("abs_tp_error:",abs_tp_error)
            print("rel_tp_error:",rel_tp_error)
            #TODO:计算 pyt_out 和 sdpa_output 之间的绝对误差总和
            abs_sp_error = ____________________________________
            #TODO:计算 pyt_out 和 sdpa_output 之间的相对误差
            rel_sp_error = ____________________________________
            print("abs_sp_error:",abs_sp_error)
            print("rel_sp_error:",rel_sp_error)
        #TODO: 计算 sdpa_output 和 tri_out 之间的绝对误差总和
        abs_ts_error = ____________________________________
        #TODO: 计算 sdpa_output 和 tri_out 之间的相对误差
        rel_ts_error = ____________________________________
        print("abs_ts_error:",abs_ts_error)
        print("rel_ts_error:",rel_ts_error)

 
if __name__ == '__main__':
    print("====================== Val =======================")
    #TODO:调用测试函数进行性能测试
    __________________________________________________________ 
     
    #print("====================== Benchmark =======================")
    #bench_flash_attention.run(save_path=".", print_data=True)

