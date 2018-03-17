# Object-oriented analysis and design 

## Unified Modeling Language

面向对象设计的特征 {
封装性（Encapsulation）：隐藏了某一方法的具体运行步骤，取而代之的是通过消息传递机制发送消息给它。
继承性（Inheritance）：在某种情况下，一个类会有“子类”。子类比原本的类（称为父类）要更加具体化。
多态（Polymorphism）：指由继承而产生的相关的不同的类，其对象对同一消息会做出不同的响应。
}

面向对象设计的原则 {
开闭原则：对扩展开放，对修改封闭
替换原则：子类可以在任何地方替换父类
依赖倒置原则：依赖关系应该是尽量依赖接口(或抽象类)，而不是依赖于具体类
接口分离原则：使用多个专门的接口比使用单一的总接口要好,使用多个专用接口比使用单一的复合接口更好
}

UML图形 {

结构性图形（Structure diagrams）强调的是系统式的建模：

* 静态图（static diagram）
        * 类图（en:Class Diagram）：a type of static structure diagram that describes the structure of a system by showing the system's classes, their attributes, operations (or methods), and the relationships among objects.
            * 类关系
        * 对象图（en:Object diagram）：a diagram that shows a complete or partial view of the structure of a modeled system at a specific time.
        * 包图（en:Package diagram）：depicts the dependencies between the packages that make up a model.
* 实现图（implementation diagram）
        * 组件图（en:Component diagram）：depicts how components are wired together to form larger components or software systems.
        * 部署图（en:Deployment diagram）：models the physical deployment of artifacts on nodes.
* 剖面图（en:Profile diagram）
* 复合结构图（en:Composite structure diagram）

行为式图形（Behavior diagrams）强调系统模型中触发的事件：
* 活动图（en:Activity diagram）：graphical representations of workflows of stepwise activities and actions with support for choice, iteration and concurrency, show the overall flow of control.
* 状态图（en:State Machine diagram）
* 用例图（en:Use Case Diagram）：a representation of a user's interaction with the system that shows the relationship between the user and the different use cases in which the user is involved.
交互性图形（Interaction diagrams），属于行为图形的子集合，强调系统模型中的资料流程：

* 通信图（en:Communication diagram）
* 交互概述图（en:Interaction overview diagram，UML 2.0）
* 时序图（en:Sequence diagram，UML 2.0）
* 时间图（en:Timing Diagram，UML 2.0）
}

类关系 {
泛化（Generalization）：即继承的反方向，指的是一个类（称为父类、父接口）具有另外的一个（或一些）类（称为子类、子接口）的共有功能。在Java中此类关系通过关键字extends明确标识。
实现（Realization）指的是一个class类实现interface接口（可以是多个）的功能。在Java中此类关系通过关键字implements明确标识。
依赖（Dependency）可以简单的理解为一个类A使用到了另一个类B，" ... uses a ..."，被依赖的对象只是作为一种工具在使用，而并不持有对它的引用，依赖关系总是单向的。在Java中体现为局部变量、方法中的参数、对静态方法的调用。
关联（Association）代表一个家族的联系。在语义上是两个类之间、或类与接口之间一种强依赖关系，是一种长期的稳定的关系，" ... has a ..." 。某个对象会长期的持有另一个对象的引用，关联的两个对象彼此间没有任何强制性的约束，只要二者同意，可以随时解除关系或是进行关联，它们在生命期问题上没有任何约定。被关联的对象还可以再被别的对象关联，所以关联是可以共享的。在Java中关联关系是使用实例变量实现的。
聚合（Aggregate）是表示整体与部分的一类特殊的关联关系，是“弱”的包含（" ... owns a ..." ）关系，成分类别可以不依靠聚合类别而单独存在，可以具有各自的生命周期，部分可以属于多个整体对象，也可以为多个整体对象共享（sharable）。表现在代码层面，和关联关系是一致的，只能从语义级别来区分。 聚合关系也是使用实例变量实现的，从Java语法上是分不出关联和聚合的。关联关系中两个类是处于相同的层次，而聚合关系中两不类是处于不平等的层次，一个表示整体，一个表示部分。
组合（Composition）关系，是一类“强”的整体与部分的包含关系（" ... is a part of ..."）。成分类别必须依靠合成类别而存在。整体与部分是不可分的，整体的生命周期结束也就意味着部分的生命周期结束。
}


# UML中几种类间关系：

泛化、实现、依赖、关联、聚合、组合

## 泛化关系/继承（Generalization)

指的是一个类（称为子类、子接口）继承另外的一个类（称为父类、父接口）的功能，并可以增加它自己的新功能的能力，继承是类与类或者接口与接口之间最常见的关系；在Java中此类关系通过关键字extends明确标识，在设计时一般没有争议性；

## 实现关系（Implementation)

指的是一个class类实现interface接口（可以是多个）的功能；实现是类与接口之间最常见的关系；在Java中此类关系通过关键字implements明确标识，在设计时一般没有争议性；

## 依赖关系(Dependence)

可以简单的理解，就是一个类A使用到了另一个类B，而这种使用关系是具有偶然性的、、临时性的、非常弱的，但是B类的变化会影响到A；比如某人要过河，需要借用一条船，此时人与船之间的关系就是依赖；表现在代码层面，为类B作为参数被类A在某个method方法中使用；

## 关联关系（Association）

他体现的是两个类、或者类与接口之间语义级别的一种强依赖关系，如客户和订单，每个订单对应特定的客户，每个客户对应一些特定的订单，再如篮球队员与球队之间的关联；这种关系比依赖更强、不存在依赖关系的偶然性、关系也不是临时性的，一般是长期性的，而且双方的关系一般是平等的、关联可以是单向、双向的；表现在代码层面，为被关联类B以类属性的形式出现在关联类A中，也可能是关联类A引用了一个类型为被关联类B的全局变量；

## 聚合关系（Aggregation）

聚合是关联关系的一种特例，他体现的是整体与部分、拥有的关系，即has-a的关系，此时整体与部分之间是可分离的，他们可以具有各自的生命周期，部分可以属于多个整体对象，也可以为多个整体对象共享；比如计算机与CPU、公司与员工的关系等；表现在代码层面，和关联关系是一致的，只能从语义级别来区分；

## 组合关系（Composition）

组合也是关联关系的一种特例，他体现的是一种contains-a的关系，这种关系比聚合更强，也称为强聚合；他同样体现整体与部分间的关系，但此时整体与部分是不可分的，整体的生命周期结束也就意味着部分的生命周期结束；比如你和你的大脑；表现在代码层面，和关联关系是一致的，只能从语义级别来区分；


>对于泛化（or继承）、实现这两种关系没多少疑问，他们体现的是一种类与类、或者类与接口间的纵向关系；其他的四者关系则体现的是类与类、或者类与接口间的引用、横向关系，这几种关系都是语义级别的，所以从代码层面并不能完全区分各种关系；但总的来说，后几种关系所表现的强弱程度依次为：依赖<关联<聚合<组合；

