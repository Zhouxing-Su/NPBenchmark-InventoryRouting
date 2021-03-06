# Note

- [ ] 初始方案
  - [x] 将一个界比较紧的 TSP 模型加入库存问题模型估算路由开销
    - [ ] 松弛布尔变量可以取实数 (完整模型)
    - [x] **不考虑子回路 (传统模型)**
  - [ ] 简单估算路由开销
    - [ ] 用直接往返于仓库与客户节点的距离作为路由开销估计
  - [ ] 将最小生成树模型加入库存问题模型估算路由开销
      - [ ] 松弛布尔变量可以取实数
      - [ ] 度约束的最小生成树
  - [ ] 简单启发信息
    - [ ] 经济配送量 (算例规律: 容量为每周期消耗的 2 到 3 倍, 初始库存为每周期消耗的 1 到 2 倍)
      - [ ] 每周期一旦有配送则配送量大于给定值
    - [ ] 经济配送次数
      - [ ] 每个客户被访问过周期的总数小于给定值
      - [ ] 每辆车每个周期访问的客户数小于给定值
  - [ ] **伪列生成**
    - [ ] 预先生成大量单个周期的 TSP 最优解
      - [ ] 从曾经找到过的次优解中学习和挖掘搜索过程中可能需要计算 TSP 的节点集合
      - [ ] 通过松弛模型等技巧直接分析搜索过程中可能需要计算 TSP 的节点集合
    - [ ] 直接通过组合最优环路求解
  - [ ] 合并各周期的需求和车容量求松弛解, 在其基础上修复
	- [ ] **准确的 TSP 近似模型**
  - [ ] **松弛部分约束使用贪心算法产生优度足够的不可行解, 然后针对可行性修复进行树搜索**
    - [ ] 松弛仓库产量与拖车容量, 尽量晚配送
- [ ] 方案调整
  - [ ] **在松弛解的基础上, 固定配送量精确计算各周期的路由得到完整解**
    - [x] 在模型的框架下, 更新最优解并禁掉当前解继续搜索, 松弛解比历史最优解差时证明最优性
      - [x] ~~惰性约束~~ vs 更新模型
      - [x] ~~禁止各周期走相同的路径~~ vs 禁止各周期访问的节点完全一致 (访问过的 + (1 - 没访问的) < N)
      - [x] 使用缓存避免重复计算单个周期的 TSP (访问的节点完全一致)
      - [ ] 使用松弛解快速修复得到的解作为 TSP 初始解 vs 使用节点集相似度较高的解作为 TSP 初始解
    - [ ] 对当前解进行分支限界, 尝试翻转各周期各节点的配送状态
      - [ ] **优度优先搜索**
        - [ ] 优先对修复后的解最好的节点分支
        - [ ] 为避免过于短视, 每次向下搜索多层再寻找最优节点
      - [ ] 优先固定确定要送或者确定不用送的
      - [ ] 限制搜索深度, 退出后从最优解出发重新开始搜索
        - [ ] 最多翻转 k 个决策变量
      - [ ] **目标函数不是确定分支顺序的唯一标准**
        - [ ] 前驱后继离当前客户很远的优先不访问
        - [ ] 给单个客户的配送量很少优先不访问, 配送时离断供还很早优先不访问
        - [ ] 仓库存储成本高客户存储成本低时应多配送, 仓库成本低客户存储成本高时应少配送
      - [ ] **多层连续翻转**
        - [ ] 删除一个客户后优先考虑在后续周期增加
        - [ ] 如果当前解不可行则优先增加节点至可行
        - [ ] 每次尝试一个客户各周期的所有配送方案组合
      - [ ] 剪枝策略
        - [ ] 被删除节点后续周期无论如何配送均会断供
  - [ ] 修改配送量目标权重得到新的配送方案, 再各周期分别计算路由
    - [ ] **使用强化学习得到路由开销的估算函数, 代入库存模型的目标得到近似完整模型求解**
    - [ ] **提升总是出现在解/子回路中的节点/边的权重 (扰动)**
    - [ ] **根据节点在修复后的最优路径中的出入度边长调整节点权重 (解决松弛子回路后模型无法准确估算下界的问题)**
    - [ ] 增加路由开销较大的周期的惩罚系数
      - [ ] 总运输成本
      - [ ] 单位运输成本
    - [ ] 增加各周期中较长边及相关节点的惩罚系数
      - [ ] 绝对长度较长
      - [ ] 远近程度排名靠后 (某节点最长出度/某节点最长入度)
  - [ ] 直接调整路由, 然后计算当前路由下的最优配送量
    - [ ] 加节点, 删节点, 周期间移动或交换节点等简单邻域动作
    - [ ] **Ejection Chain**
  - [ ] 路由确定的情况下用最小费用流计算存储成本最小的配送量
- [ ] 启发信息
  - [ ] 计算一次 N 辆车的 VRP(-TW), 将客户划分为多个区域 (局部搜索调整路由时优先考虑将同区域的节点加入路径)
    - [ ] 时间窗可以根据从初始库存或者满库存开始的断供时间设置
    - [ ] 车辆数根据大多数客户的断供时间设置 (例如大多数客户会在 3 个周期内断供, 则可以认为每 3 个周期所有客户都要被访问一次)
- [ ] 其他加速策略
  - [ ] **仅在 gap 已经比较小的情况下精确计算并添加惰性约束**
