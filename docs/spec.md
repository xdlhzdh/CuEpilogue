# 需求规格说明书：Fused GEMM + Epilogue (Fast-GELU) 自定义算子开发与编译器集成

## 1. 项目概述 (Project Overview)

本项目旨在开发一个面向 AI 芯片（NVIDIA GPU）的高性能融合算子 (Fused Operator)。核心任务是通过 CUTLASS 构建矩阵乘法与非线性激活函数（GELU）的融合，利用 Inline PTX 榨取硬件 SFU 计算单元的极限性能。最终目标是将该算子作为一个高性能后端执行体，无缝集成到基于 MLIR 等基础设施的 AI 编译器管线中，替代标准指令的 Lowering 生成。

## 2. 技术栈与运行环境 (Tech Stack & Environment)

*   **宿主语言:** C++ (C++17/C++20)
*   **计算平台:** NVIDIA GPU (Ampere 或更高架构优先)
*   **底层指令与算子库:** CUDA C++, CUTLASS 3.x (基于 CuTe), PTX ISA
*   **编译与分析工具:** NVCC, Nsight Compute (NCU), CMake
*   **上层集成目标:** AI 编译器后端 (涉及 Linalg Dialect、Bufferization 与 Code Emission)
*   **操作系统:** Linux (Fedora / Pop!_OS / Ubuntu)

---

## 3. 阶段实施拆解 (Phased Implementation Plan)

### 阶段一：Baseline 构建与内存瓶颈定位 (Native CUDA)

*   **目标:** 提供性能和数值精度的基准对照组。
*   **实现细节:** 
    *   编写标准的 Native CUDA SGEMM (单精度浮点矩阵乘)。
    *   引入基本的 Tiling 策略利用 Shared Memory，观察 Global Memory 到 Registers 的数据搬运代价。
*   **验收标准:** 输出 `nsys` 和 `ncu` 的 Profile 报告，记录核心算子的 Occupancy、访存带宽，并与理论峰值进行对比。

### 阶段二：算子融合与核心重构 (CUTLASS Integration)

*   **目标:** 利用 CUTLASS 替代 Native CUDA，实现 Device-level 算子融合，消除中间张量的显存读写瓶颈。
*   **实现细节:**
    *   **Data Type & Layout:** 定义操作数为 FP16/FP32，明确 Row-Major 或 Column-Major 的内存排布。
    *   **Threadblock & Warp Tiling:** 通过 CUTLASS 模板参数精准配置 `ThreadblockShape` 和 `WarpShape`，触发底层硬件的最优 MMA（Matrix Multiply-Accumulate）指令。
*   **验收标准:** 生成的 PTX 汇编中必须出现对齐的 `mma.sync.aligned` 指令，且计算吞吐量显著优于阶段一。

### 阶段三：极速尾声函数注入 (Inline PTX Fast-GELU)

*   **目标:** 绕过 NVCC 编译器的冗余展开，利用底层 PTX 硬件近似指令实现纳秒级的 GELU 融合。
*   **实现细节:** 
    *   基于 GELU 的代数近似公式：
        $$ \text{GELU}(x) \approx \frac{x}{1 + 2^{-2.455492 \cdot x}} $$
    *   **代码注入（两条并存路径，同一套 SFU PTX）：**
        1. **Functor（默认，Sm70 / 阶段四 runtime）：** 在 CUTLASS 2.x 风格 Epilogue Output Op（`FastGeluLinearCombination`，替换 `LinearCombination`）中嵌入 Inline PTX。
        2. **Visitor / EVT（旁路，Hopper）：** 通过 CUTLASS 3.x `CollectiveBuilder` + `epilogue::fusion::LinCombEltAct<FastGelu>`（底层 `Sm90EVT` / `FusionCallbacks`）注入同一激活。需 `-DCU_EPILOGUE_ENABLE_SM90_VISITOR=ON`。编译时 nvcc 目标架构须为 `sm_90a`（plain `sm_90` 会生成空 stub）；在 GPU 上做数值 correctness 则需 Hopper（compute capability ≥ 9.0），否则 correctness 会 SKIP（仍可做编译与静态 PTX/SASS 检查）。

```cpp
struct FastGeluPTX {
    __device__ __forceinline__ float operator()(float x) const {
        float res;
        const float constant = -2.455492f;
        asm volatile (
            "{ \n\t"
            " .reg .f32 t, r, e, p; \n\t"        // 声明 PTX 虚拟寄存器
            " mul.f32 t, %1, %2; \n\t"          // t = x * constant
            " ex2.approx.f32 e, t; \n\t"        // 硬件级快速指数近似
            " add.f32 r, e, 1.0; \n\t"          // r = e + 1.0
            " rcp.approx.f32 p, r; \n\t"        // 硬件级快速倒数近似
            " mul.f32 %0, %1, p; \n\t"          // res = x * p
            "} \n\t"
            : "=f"(res)
            : "f"(x), "f"(constant)
        );
        return res;
    }
};
```

- **验收标准:** 静态 PTX/SASS（或 NCU）中确认 SFU 指令 `ex2.approx` / `rcp.approx`（及对应 SASS `MUFU.*`）存在；数值误差在 AI 推理允许的 $\epsilon$ 范围内。Visitor 路径的编译架构与 GPU 要求见上文实现细节。

### 阶段四：AI 编译器后端集成 (Compiler Backend Emission)

- **目标:** 验证该算子在端到端系统级开发中的可用性，完成从高级中间表达 (IR) 到高性能底座的映射。
- **实现细节:**
  - **图层匹配 (Pattern Matching):** 在编译器前端/中端识别到相连的 `linalg.matmul` 与对应的非线性激活 Ops 时，进行 Pattern 融合。
  - **动态链接库封装:** 将上述 CUTLASS + PTX 算子编译为高度优化的 `.so` 动态库。
  - **代码生成 (Code Emission):** 拦截常规的 Lowering 管道（即不将其逐步 Lower 到 LLVM IR 的基础标量指令），在 Bufferization 分配好内存句柄后，直接 Emit 外部函数调用 (External Call) 指向该定制的 Fused Operator。
- **验收标准:** 编译器前端输入 Fused GEMM + GELU 的计算图，能够成功编译并执行，端到端延迟对比基础指令展开下降 30% 以上。

## 4. 关键技术风险与排查路径 (Risk Mitigation)

1. **寄存器溢出 (Register Spilling):**
   - **风险:** PTX 虚拟寄存器映射到硬件物理寄存器时超限，导致数据溢出至 Local Memory。
   - **排查:** 审查 PTX 输出，监控 `ld.local` / `st.local` 频次。必要时通过 CUTLASS 模板调小 Tiling Size。
2. **访存合并失败 (Uncoalesced Memory Access):**
   - **风险:** Epilogue 阶段写回 Global Memory 时未能达成 128-byte 事务。
   - **排查:** 结合 SASS 分析 `ld.global` / `st.global` 指令向量化状态，确保 Layout 配置与内存对齐。
3. **编译器 IR 语义不对齐:**
   - **风险:** 上层 IR（如 StableHLO / Linalg）的张量 Shape 和步长 (Stride) 与底层 CUTLASS 算子接收的 `TensorRef` 不兼容。
   - **排查:** 在 Bufferization 阶段严格实施 Shape 校验与 Padding 机制。
