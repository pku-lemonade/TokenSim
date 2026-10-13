# 剩余4.4/4.5/5.2的校准与验收契约

用户要求继续完成剩余三项。沿用tiny Qwen3 MoE、GPU2/3、BF16 eager、1A1F decode范围，不扩展prefill或GPU算子优化。旧冻结包及实验保持不变。

## 已取得的独立证据

既有18项host与7项完整操作参数冻结后，用batch6各三个独立窗口验证。25项均满足max/min<10，最差窗口倍率2.556；不因结果改变参数。对应`afd-components-independent-20261012.json`。

既有配对NCCL设备区间经覆盖审核派生transport候选：仅使用batch1/8的零延迟三窗口中位数均值，控制8-byte/int64与179-byte/uint8独立，数据512–4096-byte/BF16按方向线性插值。只计双方kernel均已启动后的服务，排除CPU提交和启动前peer等待；不是物理线缆时间。batch6在候选冻结后运行，36组同scope匹配最大倍率1.040，0/5/20ms对端延迟稳定。profiled与无profiler外层交换区间分别保留为诊断，不用于拟合。来源`afd-transport-independent-20261012.json`。

## CPU提交与设备服务的覆盖修正

前版计算组仅计函数wall减去全部CUDA API并集。新增两批API名称/拷贝类型诊断，核对原训练六窗口的原函数哈希及调用次数逐组一致。白名单仅含只读设备/occupancy查询、非阻塞kernel提交和D→D异步拷贝；同步、未知API、H→D/D→H拷贝一律拒绝。原训练窗口在函数计时前完成前序设备工作，计时只到API返回，没有完成同步。新诊断的计时不直接用于拟合。

host profile v4使用`host_submission_service`，计算组按原六窗口完整CPU调用wall的各组/窗口中位数等权拟合，F整个MoE每层仍只计一次。它替换原framework-only条目，不另行叠加；其它host和operation参数不变。CUPTI回调扰动及CPU与kernel逐条交错的聚合近似仍保留说明，不宣称精确CPU busy时间。

v4不得与旧含义不清的启动假设共存。另采当前原生GPU计算batch1/8各三个窗口，仅使用GPU kernel服务，不使用CPU、NCCL占用或整步差额；排除已包含在输入准备操作中的slot mapping。对同路由解析模型的固定设备服务系数作非负相对最小二乘，12个A/F窗口点得到4.580234514µs。这是模型的GPU固有服务偏移系数，不是物理CPU launch时延；训练最大倍率1.230。原`kernel_launch_us`参数槽位由这份明确的device_service来源替换，host v4绑定其内容哈希；不匹配的旧设备来源拒绝运行。该系数仍为聚合近似，最终需独立device同scope验收。

## 新冻结与完成条件

新增profile/schema v4预留batch7。batch1/8为训练回验，batch4为历史回归；各三组正式无hook/无profiler整步窗口，batch7另采真实路由。还需冻结后batch7三窗口GPU分项检查。已观测batch5/6不改称本轮新留出，不用留出重新选系数。

冻结源码、设备/host/operation/transport参数、输入路由与证据哈希后运行。逐步核对18项host、7项operation及6条消息真实计费，明确等待由依赖/FIFO产生，不能进入常数。GPU、transport、host/operation和整步分别按同scope比较，不能把不同窗口的服务相加构造原生step。

4.4以合格样本、参数统计与覆盖闭合验收；4.5以通用执行器实际加载、冲突拒绝与逐步计数验收；5.2以独立分项、完整step、原始/未拟合/校准后对照及相关回归验收。任一未通过不得勾选。数量级阈值仍为max/min<10，不代表小百分比误差或跨模型泛化。
