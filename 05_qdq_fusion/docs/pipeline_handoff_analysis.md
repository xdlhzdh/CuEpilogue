# INT8 QDQ GEMM：Prologue、Mainloop、Epilogue 怎么衔接

算子：

\[
D = Q\big(\mathrm{FastGelu}(s_a s_b \cdot (\textstyle\sum_k A_{mk} B_{kn} - z_a\,\mathrm{sumB}[n] - z_b\,\mathrm{sumA}[m] + z_a z_b K) + \mathrm{bias}[n])\big)
\]

\(A,B,D\) 是 int8，累加是 int32，\(s,z\) 是每个张量一个 scale 和一个 zero-point。

阅读方式：

- 第一次读：第 1–3 节。看懂三个阶段各自交出什么。
- 面试准备：第 3–5 节。能讲清三次交接，以及编译器决定改哪一段。
- 对照源码：附录。CUTLASS v3.5.1，结论与 `build/` 里的 SASS 一致。

默认实现是 V100 上的 SIMT `dp4a` kernel（[`qgemm_bias_gelu.cu`](../kernels/qgemm_bias_gelu.cu)）。Hopper 的 TMA + `wgmma` kernel（[`qgemm_bias_gelu_visitor_sm90.cu`](../kernels/qgemm_bias_gelu_visitor_sm90.cu)）只用来对照「同一份交接合同，换了搬运和计算指令」。

---

## 1. 三个阶段各管什么

一个线程块算输出 \(D\) 的一块 tile（默认 128×128）。沿着 K 把这块算完，分成三段。

| 阶段 | 做什么 | 开始时手里有什么 | 结束时交出去什么 |
| --- | --- | --- | --- |
| **Prologue** | K 循环开始前，把前几个 A/B tile 从全局内存搬进共享内存 | 全局内存里的 int8 A、B | 共享内存里已就绪的 A/B tile |
| **Mainloop** | 用共享内存里的 A/B 做 int8×int8→int32，同时去取下一个 K tile | 共享内存中的 A/B | 每个线程寄存器里的 int32 累加器 |
| **Epilogue** | 把累加器变成最终的 int8 \(D\)：反量化、bias、FastGelu、再量化、写回 | 寄存器里的 int32 累加器 | 全局内存里的 int8 \(D\) |

三段之间只有两个交接点。Prologue 不把数据直接交给 Epilogue。

```text
全局内存 A,B
    │  Prologue：预取
    ▼
共享内存 A/B tile          ← 交接点 1
    │  Mainloop：int8 乘加，同时预取下一块
    ▼
寄存器 int32 累加器        ← 交接点 2（最关键）
    │  Epilogue：布局转换 + 融合计算 + 写回
    ▼
全局内存 D
```

scale、zero-point、bias 不经过 Prologue 和 Mainloop。它们在 Epilogue（以及 Epilogue 之前的两个求和 kernel）里才出现。Mainloop 是纯整数乘加。

还有一组容易混的词：把 K 循环做成多级流水之后，循环**开头**多预取的那几级也叫 prologue，循环**收尾**把流水排空也叫 epilogue。那是 Mainloop 内部的流水调度。本文的 Epilogue 专指 K 循环全部结束后、对累加器做的后处理。

---

## 2. 先记住三个词

后面三次交接都要用到它们。

**Tile。** 全局矩阵太大，一个线程块只算输出的一小块，同时只拿 A 的几列、B 的几行。默认线程块 tile 是 \(128 \times 128\)，每次沿 K 取 32。

**累加器。** \(\sum_k A_{mk}B_{kn}\) 的 int32 部分和。K 还没走完时它一直留在寄存器里，Mainloop 结束时它就是 Epilogue 的输入。默认每个线程 64 个 int32。

**两种布局。** 同一批数，在寄存器里的排列可以不同。

- **MMA 布局**：方便从共享内存喂给 `dp4a` / `wgmma`。一个线程手里的 64 个数在输出矩阵上是散开的（默认是 2×2 个 4×4 小块，行间隔 16、列间隔 32）。
- **输出布局**：方便按行连续写回全局内存。相邻线程写相邻的列。

Mainloop 用前一种，写回要用后一种。Epilogue 负责换过去。

---

## 3. 三次衔接

### 3.1 交接点 1：Prologue → Mainloop，介质是共享内存

Mainloop 的乘加要反复读同一小块 A/B。共享内存比全局内存快，所以 Prologue 先把第一批 K tile 放进去，Mainloop 才能开工。

V100 上的具体动作（`MmaPipelined`，固定 2 级缓冲）：

1. 每个线程用 32-bit `ldg` 把全局内存里的 int8 读进寄存器。
2. 再用 `sts.128` 写进共享内存。一次写 16 字节，正好是 K 方向连续的 4 个 int8 打包成一个 32-bit 字。
3. `__syncthreads()`。这一步是交接本身：写的线程和马上要读的线程必须看到同一块共享内存。

共享内存里 A 用 `ColumnMajorInterleaved<4>`，B 用 `RowMajorInterleaved<4>`。这样排是为了让一条 `dp4a` 的一个操作数正好占一个寄存器。布局是从计算指令倒推出来的。

Mainloop 跑起来之后，Prologue 的工作并入循环：算第 \(k\) 块时，把第 \(k+1\) 块先读进寄存器；这一轮乘加和同步结束后，写入另一半共享内存。两块缓冲轮流当「正在算」和「正在装」。V100 没有 `cp.async`，全局内存到共享内存必须经寄存器中转。

Hopper 上交接合同不变，介质还是共享内存，搬运改成另一个 warp group 发 TMA（`UTMALDG`），计算 warp group 不再自己做加载。

### 3.2 交接点 2：Mainloop → Epilogue，介质是寄存器

这是整条管线的断层。

Mainloop 最后一轮把共享内存里的 A/B 乘完之后，**部分和已经全部在寄存器里**。共享内存里的 A/B 从此不再被读。Epilogue 的输入就是这组 int32。

默认形状：线程块 128×128，warp 32×64，8 个 warp、256 个线程。一个 warp 内线程排成 4（行）×8（列），每线程一块 8×8，共 64 个 int32。SASS 里一个 K=32 的 tile 对应 `IDP.4A.S8.S8` × 512，即 64 个累加器 × 8 条 `dp4a`。

这 64 个数是 MMA 布局。Epilogue 要按行写 \(D\)，还要按列取 `bias[n]`、`sumB[n]`，按行取 `sumA[m]`，所以要知道每个寄存器对应的 \((m,n)\)。

V100 的做法是先换布局，再计算（`epilogue/threadblock/epilogue.h`）：

1. 把一部分累加器按 MMA 布局写进共享内存。
2. `__syncthreads()`。
3. 按输出布局读回：相邻线程拿到相邻列。
4. 对读回的数做融合计算，写回全局内存。
5. 64 个数分 8 轮做完。

第 1 步写的那块共享内存，就是 Mainloop 刚用完的 A/B 缓冲区。`kernel::Gemm` 里两者是一个 `union`：Mainloop 的缓冲区在最后一次同步之后已经死了，Epilogue 的转置区这才开始活，活跃区间不重叠，可以占同一块地址。

Hopper 的交接介质同样是寄存器，但**不把累加器搬去共享内存**。`wgmma` 的累加器保持原布局，融合计算直接在寄存器里做（`cst_callbacks.visit`）。共享内存只暂存另外两样东西：TMA 读进来的 C，以及算完后准备 TMA 写出的 D。

两种做法的差别可以收成一句：逐元素运算（乘 scale、GELU、量化）在哪种布局下结果都一样；**只有需要 \((m,n)\) 的读取才迫使你要么换布局，要么在 MMA 布局下算出坐标**。

### 3.3 交接点 3：Epilogue → 全局内存

V100 上，融合计算全部在 `AffineGeluQuantEpilogue::applyOne` 里，对一个 int32 累加器串起来：

1. 用调用次数和输出线程映射反推出 \((m,n)\)。
2. int32 上做零点修正：减去 \(z_a \mathrm{sumB}[n]\)、\(z_b \mathrm{sumA}[m]\)，加回 \(z_a z_b K\)。
3. 转成 float，乘 \(\alpha = s_a s_b\)，加上 `bias[n]`。
4. `FastGeluPTX`：`ex2.approx.f32` 和 `rcp.approx.f32`。
5. 乘 \(1/s_d\)，加 \(z_d\)，取整并饱和到 \([-128,127]\)。
6. 写一个 int8。

写回是标量 `STG.E.U8`。CUTLASS 的 SIMT `dp4a` 特化要求 Epilogue 每次处理 1 个元素，所以没有 128-bit 向量写。合并访存靠线程映射：一个 warp 的 32 个线程写 32 个连续字节。

Hopper 上同一条数学式由 visitor 树完成：先 \(\beta C + \alpha \cdot \mathrm{acc}\)，再 `FastGeluQuant`。结果用 `STS.U16` 写入共享内存（一次两个 int8），再由 TMA（`UTMASTG`）整块写回。`AlignmentD = 16` 个 int8，对齐到 128 bit，这是 TMA 对全局地址的要求。

### 3.4 对照

| | V100 SIMT `dp4a` | Hopper TMA + `wgmma` |
| --- | --- | --- |
| 交接点 1 | 本线程 `ldg` → 寄存器 → `sts`，2 级缓冲 | producer warp group 发 TMA，多级缓冲 |
| 交接点 2 | 累加器写共享内存，按输出布局读回，再计算 | 累加器留在寄存器里直接计算 |
| 交接点 3 | 每线程写 1 字节 | 共享内存暂存 D，TMA 写出 |
| 共享内存 | Mainloop 与 Epilogue 是 `union`，分时复用 | 两块 `struct` 并列，同时活着 |

Hopper 不能复用的原因：它是持久 kernel。producer 给下一个 tile 装 A/B 的时候，consumer 还在跑当前 tile 的 Epilogue，两段活跃区间重叠。`StageCountAutoCarveout` 先从共享内存容量里扣掉 Epilogue 那一块，剩下的才分给 Mainloop 的级数。Epilogue 越占共享内存，Mainloop 能重叠的预取就越少。

---

## 4. 为什么量化参数出现在 Epilogue

把 DQ 提前乘进矩阵乘，得到：

\[
\sum_k s_a(A_{mk}-z_a)\,s_b(B_{kn}-z_b)
= s_a s_b\Big(\sum_k A_{mk}B_{kn} - z_a\,\mathrm{sumB}[n] - z_b\,\mathrm{sumA}[m] + z_a z_b K\Big)
\]

左边每个乘加都是浮点，int8 的 `dp4a` / `wgmma` 用不上。右边的求和是纯 int8 乘加，scale 和零点修正每个输出元素只做一次。

能这样提的条件：**scale 和 zero-point 沿 K 不变**。

| 量化方式 | 能否放到 Epilogue |
| --- | --- |
| 每个张量一个 scale（本仓库） | 能。\(\alpha\) 是标量 |
| A 按行、B 按列（per-token / per-channel） | 能。Epilogue 多读 `s_a[m]`、`s_b[n]` |
| 沿 K 分组（GPTQ / AWQ 那种 group-wise） | 不能。scale 随 \(k\) 变，必须在 Mainloop 里、乘加之前把 B 反量化 |

零点修正要整行的和、整列的和，一个输出 tile 内部算不出来，所以 GEMM 之前另有 `RowSumKernel` 和 `ColSumKernel`。

两条路径把修正项送进 Epilogue 的方式不同：

- V100：Epilogue 里按 \((m,n)\) 读 `sumA[m]`、`sumB[n]`、`bias[n]`，当场算。
- Hopper：`FillAffineCKernel` 先把「零点修正 + bias」写成一整张 \(M \times N\) 的 fp32 矩阵 C，Epilogue 用 \(\beta=1\) 加回来。少写自定义 visitor，多出约 \(8MN\) 字节访存。

---

## 5. 编译器改的是哪一段

本仓库的编译链不生成 kernel，它把一串算子收成对一个已经写好的 kernel 的调用。

```text
linalg: DQ, matmul, bias, FastGelu, Q
    │  fuse-qdq-qgemm-bias-gelu     FuseQdqPattern.cpp
    ▼
cutlass.qgemm_bias_gelu
    │  one-shot-bufferize           必须发生在融合之后
    ▼
同一个 op 的 memref 形式
    │  lower-cutlass-qgemm-to-call  LowerToCall.cpp
    ▼
func.call @cutlass_qgemm_bias_gelu
```

融合必须在 bufferize 之前。否则 DQ 和 GELU 的 fp32 中间结果会被分配成 \(M \times K\)、\(M \times N\) 的缓冲区。

图上的每个算子落到 kernel 的固定阶段：

| 图上的算子 | 落到 |
| --- | --- |
| DQ | 拆开。int8 进 Mainloop；scale 变成 Epilogue 的 \(\alpha\)；zero-point 变成 Epilogue 的修正项，外加 GEMM 前的 RowSum / ColSum |
| matmul | Mainloop |
| bias，下标 \((i,j)\rightarrow(j)\) | Epilogue，按列读取 |
| FastGelu | Epilogue 的激活函数 |
| Q | Epilogue 的最后一步，写出 int8 |

匹配条件就是 kernel 的能力边界。`matchBiasAdd` 只接受按列广播，因为 Epilogue 只实现了按列取 bias。scale 必须是标量，因为第 4 节的提取只对 per-tensor 实现了。条件放宽会生成算错的代码，收紧会漏掉能融合的图。

### 改 Epilogue 还是改 Mainloop

先问：新运算依赖的量，沿 K 变不变？

- **沿 K 不变**（per-channel scale、另一种激活、按行或按列的 bias）：只改 Epilogue 和它前面的模式匹配。Mainloop 不动。
- **沿 K 变**（group-wise 量化）：改 Mainloop，在共享内存读到寄存器之后、乘加之前做反量化。Epilogue 不动。

以「B 改成 per-channel，`scale_b` 变成长度 N 的向量」为例，改动顺序是：

1. 确认 `scale_b[n]` 沿 K 不变，可以留在 Epilogue。
2. `matchDequant` 接受下标 \((k,n)\rightarrow(n)\) 的 scale，不再只接受标量。
3. `cutlass.qgemm_bias_gelu` 和 C ABI 增加一个指针。
4. Epilogue 里 \(\alpha\) 从标量改成 \(s_a \cdot \mathrm{scale\_b}[n]\)。

### Epilogue 本身有三种改法

| 做法 | 能表达的运算 | 本仓库 |
| --- | --- | --- |
| 换 CUTLASS 2.x 的 OutputOp | 逐元素。接口不提供 \((m,n)\) | V100。`coordsForCall` 用调用次数反推坐标 |
| 生成 Epilogue Visitor Tree | 逐元素、按行/列广播、再读一张表 | Hopper 只用了现成的 `LinCombEltAct` |
| 自己生成 K 循环后面的 IR | 任意 | 未做 |

V100 的反推坐标能工作，前提是一次只处理 1 个元素、swizzle 是 identity、Epilogue 的调用顺序不变。任一前提变了，坐标会静默算错。

Visitor 树才是编译器该生成的东西。第 4 节的公式可以收成

\[
y = \alpha\cdot\mathrm{acc} + r[m] + c[n]
\]

其中 \(r[m]\) 只随行变，\(c[n]\) 只随列变（bias 和 `sumB` 都折进 \(c\)）。对应的树（设计草图，没有编译过）：

```cpp
using RowTerm = Sm90ColBroadcast<0, TileShape, float>;  // r[m]
using ColTerm = Sm90RowBroadcast<0, TileShape, float>;  // c[n]
using Affine = Sm90EVT<Sm90Compute<homogeneous_multiply_add, float, float>,
                       Sm90ScalarBroadcast<float>, Sm90AccFetch, RowTerm>;
using Biased = Sm90EVT<Sm90Compute<plus, float, float>, Affine, ColTerm>;
using Fusion = Sm90EVT<Sm90Compute<FastGeluQuant, int8_t, float>, Biased>;
```

翻译是机械的：累加器 → `Sm90AccFetch`；标量 → `Sm90ScalarBroadcast`；下标 \((i,j)\rightarrow(j)\) → `Sm90RowBroadcast`；下标 \((i,j)\rightarrow(i)\) → `Sm90ColBroadcast`；逐元素算术 → `Sm90Compute`。这样 Epilogue 只多读 \(M+N\) 个 float，不再物化那张 \(M \times N\) 的 C。

落地方式是按签名生成一段 CUTLASS 实例化代码，AOT 编译并缓存。本仓库现在是一份固定的 `.so` 加一个 C ABI，相当于只缓存了一个签名。

### 和自研 kernel 编译器的对应关系

| 这里手写死的衔接 | 编译器里的 pass |
| --- | --- |
| Prologue 预取、Mainloop 多级缓冲 | 软件流水：把 K 循环的加载提前若干级 |
| 共享内存的 Interleaved&lt;4&gt; | 从乘加指令的操作数形状反推共享内存布局 |
| 累加器经共享内存换成输出布局 | 布局转换（Triton 的 `convert_layout`）。后面若全是逐元素运算，转换可以推迟到写回前 |
| `union` 或 `struct` | 共享内存分配：活跃区间不重叠就共用偏移 |

---

## 6. 当前实现的缺口

1. V100 的 \((m,n)\) 是反推的，换 swizzle 或 Epilogue 调用方式会算错。应改成能拿到坐标的 visitor。
2. Hopper 为了复用 `LinCombEltAct`，物化了 \(M \times N\) 的 fp32 C。应按第 5 节收成按行、按列的广播。
3. B 是行优先（N-major），而 8-bit `wgmma` 要求从共享内存读的操作数是 K-major。CUTLASS 因此把 B 逐字节读进寄存器再重排（`LDS.U8` 与 `PRMT`）。B 若是常量权重，应在编译期转成 K-major。
4. Hopper kernel 没有在真机上对过数。V100 上对应测试会 SKIP。内部对调 A/B 时连 M/N 一起对调，输出坐标要在 Hopper 上跑正确性测试才能确认。

---

## 附录 A. 代码在哪

| 阶段 | 本仓库 | CUTLASS v3.5.1（`include/cutlass/`） |
| --- | --- | --- |
| 图融合 / 降到调用 | [`lib/FuseQdqPattern.cpp`](../lib/FuseQdqPattern.cpp)、[`lib/LowerToCall.cpp`](../lib/LowerToCall.cpp) | — |
| V100 三段组装 | [`kernels/qgemm_bias_gelu.cu`](../kernels/qgemm_bias_gelu.cu) | `gemm/kernel/default_gemm.h`（DP4A 特化）、`gemm/kernel/gemm.h`（`union SharedStorage`） |
| V100 Prologue + Mainloop | — | `gemm/threadblock/mma_pipelined.h`、`gemm/threadblock/default_mma_core_simt.h` |
| V100 Epilogue | [`kernels/qgemm_epilogue_op.cuh`](../kernels/qgemm_epilogue_op.cuh) | `epilogue/threadblock/epilogue.h` |
| Hopper 三段组装 | [`kernels/qgemm_bias_gelu_visitor_sm90.cu`](../kernels/qgemm_bias_gelu_visitor_sm90.cu) | `gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` |
| Hopper Mainloop | — | `gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp` |
| Hopper Epilogue | [`kernels/fast_gelu_quant.cuh`](../kernels/fast_gelu_quant.cuh) | `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`、`epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` |
| FastGelu | [`03_fastgelu_epilogue/fast_gelu_ptx.cuh`](../../03_fastgelu_epilogue/fast_gelu_ptx.cuh) | — |

## 附录 B. SASS 里看到的衔接

`cuobjdump -sass` 只统计 GEMM kernel 本身。

| 衔接 | V100（sm_70） | Hopper（sm_90a） |
| --- | --- | --- |
| 交接点 1 | `LDG.E` → `STS.128` | `UTMALDG` |
| Mainloop | `IDP.4A.S8.S8` × 512 | `IGMMA.64x128x32.S8.S8`，一个操作数来自寄存器 |
| 交接点 2 | Epilogue 内 `STS` / `LDS`，夹在两次 `BAR.SYNC` 之间 | 累加器不进共享内存；B 的重排是 `LDS.U8` × 192、`PRMT` × 176 |
| 融合计算 | `MUFU.EX2`、`MUFU.RCP` | 同左，另有 `I2FP.F32.S32`、`F2IP.S8.F32` |
| 交接点 3 | `STG.E.U8` | `STS.U16` → `UTMASTG` |
