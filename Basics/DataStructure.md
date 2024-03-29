```
数据结构三要素：{
逻辑结构：线性、非线性
存储结构：顺序、链式、索引、散列
数据运算
}
程序复杂度{
定义：用来衡量算法运行所需要的计算机资源（时间、空间）的量
时间复杂度：指算法中所有语句的频度（执行次数）之和，一般以它最深层的循环的数量级来表征
空间复杂度：辅助空间大小
}
函数渐进阶{
定义：T(n)表示某个给定算法的复杂度。所谓渐进性态就是令n→∞时，T(n)中增长最快的那部分
符号{O上界[<=]，Ω下界[>=]，θ同阶[=]，o[<]}
}
线性表：{
定义：具有相同数据类型的n(n>=0)个元素的有限序列
特点：线性有序
分为顺序表（顺序存取、随机存取）和链表（顺序存取）
}
栈：仅在一端插入删除的线性表（后进先出）
队列：仅在一端插入另一端删除的线性表（先进先出）
树（数据结构）{
定义：n(n>=0)个节点组成的一个具有层次关系的集合
* 每个节点有零个或多个子节点；
* 没有父节点的节点称为根节点；
* 每一个非根节点有且只有一个父节点；
* 除了根节点外，每个子节点可以分为多个不相交的子树；
}
二叉树：每个节点至多两棵子树的有序树
满二叉：高h，含2^h-1个结点
完全二叉：与满二叉的结点一一对应
森林：m（m>=0）棵互不相交的树的集合
哈夫曼树：带权路径长度最小的二叉树
哈夫曼编码：对高频字符赋以短编码（无损数据压缩编码）
图：由点集和边集组成，G=(V, E)，V非空【树可以是空树，图不能是空图】
生成树：包含全部顶点的极小连通子图
最小生成树：{
定义：边权之和最小的生成树
算法：Prim、Kruskal
}
最短路径算法：Dijkstra、Floyd
拓扑排序：对有向无环图的排序，若A在B前，则不存在B->A的路径
稀疏矩阵存储：三元组<行,列,值>
B树{
定义：多路平衡查找树(平衡因子0)
特点：每个节点可有多个关键字(有序)，其子树个数=关键字个数+1
查找：多路分支决定。每个节点都是多关键字有序表
仅多路查找
}
B+树{
定义：B树的一种变形树
应用：通常用于数据库和文件系统中
区别：B+树中，n个关键字的结点含有n棵子树，非页结点仅起索引作用
顺序查找&多路查找
}
hash函数{
散列表：hash(key)=address
构造方法：直接定址hash(key)=a*key+b、除留余数hash(key)=key%p、
冲突处理：{
开放定址：h1=(hash(key)+di)%m（线性探测di=i、平方探测di=i^2、再散列di=h2(key)）
拉链法：将所有同义词存储在一个线性链表中
}
*各排序算法时空复杂度
```