# AFD 固定host成本与因果等待

本轮实现用户确认的固定开销近似，属于OpenSpec `align-vllm-afd-protocol-and-timing`任务8.1/8.2。准备、提交使用固定服务成本；同步和排队由当前事件与资源状态决定。实现放在通用AFExecutor，ModelAFExecutor和标准AF引擎共用，未修改远端插件或启动GPU测试。

## 配置

在已有`af.communication_timing`下添加以下四项，单位µs，默认0：

| 参数 | 计费位置与次数 |
| --- | --- |
| `a_step_prepare_us` | iteration入口，每个参与A worker一次，A计算等待完成 |
| `f_step_prepare_us` | iteration入口，每个参与F worker一次，可与A计算重叠，F接收提交等待完成 |
| `compute_submit_us` | 每个compute DAG节点的额外CPU调度/包装工作一次，不按专家数倍增，不含已有kernel启动费用 |
| `host_receive_submit_us` | 每条A→F或F→A数据消息的接收提交一次，与发送提交独立竞争host资源 |

非零值只支持现有`vllm_afd` decode支持域且要求`enabled=true`；负数、NaN、无穷大、bool均拒绝。原有`host_submit_us`、`host_completion_us`和`host_resource_scope`继续生效。新旧所有host服务共用同一per_worker/shared资源，服务结束立即释放。原有通信提交默认5µs保持不变。

固定参数是未校准假设。`compute_submit_us`不是每个kernel launch的价格：一个attention DAG节点可能含多个kernel，既有算子估值已经包含的launch成本不得重复填入。独立CPU输入准备/编码可以近似为每步成本；必须等待GPU结果的回读与metadata网络消息不能提前折入step准备。正式profile的覆盖校验与metadata协议仍待原任务完成。

## 等待与重叠

首版在消息数据依赖完成后建立接收提交，接收方同时等待本步准备；这属于按需接收近似，尚未回放原生提前预贴recv。双方提交、输入和额外readiness事件均完成后才申请fabric。host queue、peer readiness wait、fabric queue及transport service分开记录，事件busy/等待之和不能当作整步墙钟。

确定性测试：A计算100µs、F从0开始准备200µs、其他host成本0、传输5µs时，传输区间为[200,205]µs；A侧等待100µs，fabric排队0。不会把两段可重叠工作相加得到305µs。

本步准备事件每次iteration创建，每个worker一次，多block/多layer共享；不写入DAG或service缓存。compute/transport service缓存仍只保存纯服务。取消iteration会清理准备、接收提交与资源请求。trace关联本步准备、提交、transport及完成，模型跨层trace连接不会把独立F准备错误地挂到A embedding之后。

## 使用示例

从仓库根目录运行以下本地模拟命令；数值仅用于验证机制，不是PKUa100校准参数：

```powershell
python -m scripts.run_tiny_afd_alignment --output outputs/fixed-host-example --batches 1 8 4 --no-native-control --a-step-prepare-us 10 --f-step-prepare-us 200 --compute-submit-us 3 --host-receive-submit-us 2
```

报告保留`fixed_host_assumptions`与`calibration_status=uncalibrated`，不会把加入固定值后的结果标成校准成功。示例batch1的31步平均为395.322µs，仅是上述假设的输出。省略新增参数恢复原来的零新增成本基线。

## 验证与剩余工作

新增9项测试覆盖重叠oracle、完整模型与连续iteration、多个block唯一计费、shared host竞争、trace/cache等价、非法配置、取消准备、取消接收及资源复用。保留既有计算/传输估值，未拟合实测差额。

`python -m unittest discover -s tests -p 'test_af*.py' -q`：236项通过（含新增9项）；OpenSpec严格校验通过。

尚需独立完成：原生metadata消息与回读依赖、精确host调用顺序/接收预贴、compute融合覆盖、真实路由回放、纯传输测量与版本化校准profile。固定成本机制完成不代表3.2–3.7ms实测与模拟差距已经消除，原OpenSpec任务3.2等完整验收仍保持未完成。
