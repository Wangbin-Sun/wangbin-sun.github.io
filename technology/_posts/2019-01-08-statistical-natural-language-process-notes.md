---
layout: post
title: Statistical Natural Language Process (Notes)
description: >
  宗成庆《统计自然语言处理》笔记
excerpt_separator: <!--more-->
image: /assets/img/blog/statistical_natural_language_process.jpg
---
该书可作为自然语言处理的查阅资料，内容翔实。

## 第1章 绪论
* 计算语言学包括语音与语义
* 形态、语法、语义、语用等层次，核心问题都是歧义消解
  * 歧义组合可由开塔兰数（Catalan numbers）计算 
  * $$C_n = {2n \choose n}\frac{1}{n+1}$$ 其中n为介词短语数
* 自然语言处理的研究方法有理性主义和经验主义
* 许多自然语言处理的任务可以由**噪声信道模型**实现和完成

## 第2章 预备知识
* 全概率公式 $$P(A) = \sum_i P(A \vert B_i)P(B_i)$$
<!--more-->
* 通常假设一个句子出现独立于它之前的语句，句子概率分布近似符合二项式分布
* [熵]定义$$H(X) = -\sum_{x\in R}p(x)\log_2p(x)$$
  * 熵最大时，随机变量最不确定
* [熵连锁规则]$$H(X_1,..., X_n)= H(X_1) + ...+H(X_n \vert X_1, ...,X_{n-1})$$ 这里涉及到联合熵与条件熵
* [互信息]$$I(X;Y) = H(X) * H(X \vert Y)$$知道了Y的信息后，X不确定性的减少量
  * $$I(X;X) = 0$$因而熵又称为自信息
* [相对熵]$$D(p\|q) = \sum_{x\in X}p(x)log\frac{p(x)}{q(x)}$$ 两个概率分布相对差距的测度
  * 互信息是衡量联合分布与独立性的相对熵
* [交叉熵]$$H(X,q) = H(X) + D(p\|q)$$ p是真实分布，q是近似分布，X是p的随机变量
  * 困惑度常作为模型质量评估指标
* 线性分类器存在对偶形式

## 第3章 形式语言与自动机
* [闭包]$$V^*=V^++{\epsilon} = V^1\cup V^2\cup ...+V^0$$
* [规范推导]又称最右推导，每次只改写最右边的非终结符
* [文法分类]正则文法、上下文无关文法、上下文相关文法和无约束文法
* [派生树]CFG（Context-free Grammer）可表示，也称语法树、分析树、推导树
* [自动机]Automata，包含状态、转移方程等的集合变换方式
  * 正则文法 * 有限自动机
  * 上下文无关文法 * 下推自动机
  * 上下文相关文法 * 线性界限自动机（单带图灵机）
  * 无约束文法 * 图灵机
* [有限状态机]路径从初始状态到终止状态经过的所有弧上的字母连接起来构成一个字符串
* [有限状态转换机]与有效状态机区别在于完成状态转移的同时产生一个输出

## 第4章 语料库与语言知识库
* [语料库]分几个层次，除了本身包含的词汇外，还会有句法、深层语法等内容，也可以是多媒体的形式
* [语言知识库]包括语料库，分为显性结构化知识和隐性文本信息两类
* [本体论]核心概念是知识共享，内容包括主要概念以及它们之间的关系
  * 本体可分为上位本体、领域本体以及面向应用的本体三个层次 

## 第5章 语言模型
* [n元语法]将“历史”限定在前n个时间段，n=3即三元文法模型称为二阶马尔可夫链
  * $$w_i^j$$指第i位到第j位的“语法”,是j-i+1阶。 
* [交叉熵]利用预测和压缩的关系计算：$$H_p(T) = -\frac{1}{W_T}log_2p(T)$$
  * $$p(T)$$为文本T的概率，$$W_T$$是以词为单位度量的文本T的长度
  * 含义：利用与模型$$p(w_i\vert w^{i-1}_{i-n+1})$$有关的压缩算法对数据集合中的$$W_T$$个词进行编码，每一个编码所需要的平均比特位数
* [平滑]解决零概率问题，一些未出现在语料中的零概率情形实则并非零概率
  * Good-Turing估计法是很多平滑技术的核心
    * $$r = (r+1) \frac{n_{r+1}} {n_r}$$来平滑
    * $$p_r = \frac{r^*}N$$, $$N = \sum^\infty _{r=0}n_rr^*=\sum^\infty _{r=1}n_rr$$
  * Katz对于非零小计数值的语法减值，折扣率$$d_r\approx \frac{r^*}r$$
  * Jelinek-Mercer 用低阶模型差值预测高阶模型
    * Witten-Bell是一个实例
    * 绝对减值法类似差值，不过直接减去固定值来建立高阶分布
    * Knesey-Ney是拓展的绝对减值法，使用一元文法的概率应与它前面的不同单词数目成比例
  * 平滑算法分为后备模型和差值模型
* 平滑方法中，Knesey-Ney及其变种表现最好
* [自适应模型]语言模型理论基础较为完善，但需要考虑跨领域的问题
  * 基于缓存模型假设近期出现的词在后续出现可能性大
  * 基于混合模型先对训练语料聚类，对不同领域分别训练模型

## 第6章 概率图模型
* [概率图]基于图的方法表示概率分布，结点表示变量，边表示变量之间的概率关系
* [生成式模型]又称产生式模型，假设状态序列y决定观测序列x，是所有变量的全概率模型
* [区分式模型]又称判别式模型，假设观测序列x决定状态序列y，是传统的模式分类思想，通常是有监督学习
* [贝叶斯网络]两个节点有联结表示这两个随机变量在任何情况下都不存在条件独立
  * 构造包括表示、推断、学习等步骤 
* [马尔可夫模型]描述当前状态与历史n阶状态的转移关系 
* [隐马尔可夫模型]两个随机过程，状态及其转移未知，但状态的输出可观测
  * 基本问题包括估计问题、序列问题和训练问题
  * 估计/解码问题求解给定模型下，特定观测序列的概率
    * 动态规划求解，仅与上一时刻的状态相关，即前向算法
    * 同样，也可以使用后向算法来计算，也可结合前向后向同时计算
  * 序列问题求解给定模型和观测结果，最优状态序列
    * Viterbi算法可求解，利用动态规划，维特比变量类似前向变量，存在递归关系 
  * 训练问题/参数估计问题，给定观察序列，调节参数使其概率最大
    * 利用EM算法直接模拟，得到隐变量的期望值，代替待估计值 
* [层次化的隐马尔可夫模型]每个状态本身就是一个独立的HHMM，一个状态产生一个观察序列而非观察符号 
* [马尔可夫网络]类似贝叶斯网络，能额外表示循环依赖，但无法表示推导关系，是无向图模型
* [最大熵原理]在已知部分信息的前提下，关于未知分布最合理的推断应该是符合已知信息最不确定或最大随机的推断，即使熵值最大
    * 可用通用迭代算法GIS训练参数，选取有效的特征$$f_i$$ 及其权重$$\lambda_i$$
* [最大熵马尔可夫模型]结合隐马尔可夫链和最大熵模型，又称为条件马尔可夫模型
    * MEMM是有向图和无向图的混合模型，主体仍然是有向图。
    * 相比HMM，优点在于可任意选择特征
* [条件随机场]通过定义条件概率而非联合概率描述模型。CRF可看做是一个无向图模型或马尔可夫随机场 

## 第7章 自动分词、命名实体识别与词性标注
* [自动分词]针对汉语等孤立语、黏着语区分词，主要困难在于分词规范、歧义切分和未登录词的识别
    * 分词规范指对于词的抽象定义与具体界定没有得到统一
    * 歧义切分包括交集型切分歧义与组合型切分歧义
    * 未登录词又称生词
        * 大规模真实文本中，未登录词对于分词精度的影响远远超过了歧义切分 
* [汉语分词]先使用切分算法进行粗分，再进行歧义排除和未登录词识别
    * [N-最短路径]构造词语切分有向无环图，每个词对应一条有向边
    * [由字构词]转化为序列标注问题
    * [词感知机]直接使用词相关的特征，而非字相关特征
    * 基于词的生成式模型对集内词处理性能较好，基于字的区分时模型对集外词的处理更鲁棒
* ［命名实体］实体概念的引用有三种形式，命名性指称、名词性指称、代词性指称
    * ［基于CRF］将命名实体识别看作是序列标注问题
    * 实体识别的内容主要包括人名、地名以及机构名
    * 均为试图充分发现和利用实体所在的上下文特征和实体的内部特征，仅特征的颗粒度有大小
* ［词性标注］判定句子中每个词的语法范畴，确定其词性并加以标注
    * 同分词一样，是中文信息处理的重要基础性问题
    * 基于统计：HMM模型参数估计，结合数据平滑方法
    * 基于规则：按兼类词搭配关系和上下文语境建造消歧规则，可应用机器学习方法
* ［词性标注的检验］词性标注一致性指在相同的语境下对同一词标注相同的词性
* 汉语分词研究的总体水平，F1值已经达到95%

## 第8章 句法分析
* 句法分析分为句法结构分析和依存关系分析两种
* ［句法结构分析］使用句法结构分析器得到合乎语法的句子的树状数据句法结构
    * 句法结构歧义的识别和消解是面临的主要困难
    * 一部分是形式化的规则库和词典构成举法分析的知识库；另一部分是分析算法的设计
    * ［语法形式化］目前通常采用上下文无关文法（CFG）和基于约束的文法
    * ［基于规则］通常有自顶向下、自底向上、两者结合的分析方法
    * ［基于统计］通常语法驱动，由生成语法定义被分析语言及其分析出的类
* ［PCFG］是CFG的扩展，引入概率，具有位置不变性、上下文无关性和祖先无关性的特征
    * 计算结构概率，利用DP的内向外向算法
    * 选择最佳结构，利用Viterbi算法结合DP
    * 概率参数估计，采用EM迭代算法
*  ［词汇化的短语结构分析］对句法树中的每个非终结符都利用其中心词（词性）标注，通过马尔可夫过程求解
*  ［非词汇化句法分析］上下文无关的假设离现实存在距离，加入隐含标记
*  ［短语结构分析器性能评价］标记正确率、标记召回率、交叉括号数
    * 英语句子的分析准确率已超过90%
    * 汉语的性能约低5%
* ［长句结构分析］通常算法时间复杂度为$$O(N^3)$$，局部子句的错误会导致常聚得不到正确的句法关系树
    * 利用标点符号切分单元
*  ［HP算法］Hierarchical Parsing，层次分析
    *  依据标点分割长句
    *  对各子句分别句法分析
    *  第二遍分析得到子句结构关系
*  ［浅层句法分析］又称部分句法分析、语块划分，仅要求识别句子中某些结构相对简单的独立成分
    * ［基本名词短语］短语的中心语为名词，短语中不含有其他子项短语；可采用括号分隔法或IOB标注方法
    * 常用方法同样有基于统计和基于规则的，包括SVM、WINNOW、CRF
* ［依存语法］句子中词与词间的依存关系，又称为从属关系语法，词之间关系不对等，存在支配关系
    * 依存树是一颗有“根”树 
    * 依存分析方法包括生成式、判别式和确定式
        * 生成式和判别式是机器学习两类达模型的拓展应用  
        * 确定性分析策略包括“依次读入”和“立即处理”
    * 汉语依存分析性能效果较英语差许多
    * 短语结构树可对应转换成依存关系树

## 第9章 语义分析
* ［词义消歧］确定一个多义词在上下文语境中的具体含义，也称为词义标注
* ［有监督的词义消歧］利用有监督的机器学习解决词义消歧问题（分类问题）
    * ［基于互信息方法］利用Flip-Flop算法，迭代找到最佳的互信息
    * ［基于贝叶斯分类器方法］朴素贝叶斯假设词间独立，极大似然方式估计
    * ［基于最大熵方法］示例特征包括词形信息、词性信息、词形及词性信息
* ［基于词典的词义消歧］结合本身词典信息，如Yarowsky算法
* ［无监督的词义消歧］通过词义辨识进行，随机赋值并利用EM算法估计
* 词义消歧目前算法在正确率和召回率不到80%
* ［语义角色标注］浅层语义分析技术，以句子为单位，分析谓词－论元结构，即各成分与谓词的关系
    * 受句法分析的准确率影响大，每一个论元相应于句法树的某个节点
    * ［基本流程］假定谓词给定，进行句法分析，剪除候选论元，论元辨识与标注，处理后得到标注结果
    * 不同语义角色标注方法区别在于利用的句法分析树不同
    * 为了减轻句法分析错误，可以对多个语义角色标注系统的结果进行融合
    * 汉语语义角色标注F1超过70%，融合方法F1达到80%
    * 语义角色标注在领域内外测试集的F1差距一般在10％上下
* ［双语联合语义角色标注］联合推断模型可采用整数线性规划，涉及三部分包括源语言、目标语言和双语两端的论元对齐
    * 双语联合推断模型的目标函数是三个子目标函数的加权和

## 第10章 篇章分析
* 篇章分析最终目的从整体上理解篇章，核心任务分析篇章结构
    * 篇章结构包括逻辑语义结构、指代结构、话题结构
    * ［篇章的基本特征］衔接性、连贯性、意图性、信息性、可接受性、情景性和跨篇章性
* 篇章分析理论，最早为概念依存方法，提出脚本方法基于场景填入slot
    * ［言语行为理论］语言不是用来陈述事实或描述事物的，而是负载着言语者的意图；包括言内行为、言外行为和言后行为
    * ［中心理论］篇章由三个分离的但相互关联的部分组成，包括话语序列结构、目的结构和关注焦点状态
    * ［修辞结构理论］描述各部分的修辞关系来分析篇章结构和功能，提出两种篇章单位，为核心和卫星
    * ［脉络理论］建立在中心理论和修辞结构理论之上，拓展到宏观语篇，但只关心拓扑结构
    * ［篇章表示理论］篇章是自然语言理解的完整单位，构造篇章表示结构，包括篇章指称对象和与指称对象有关的条件
* ［篇章衔接性］衔接即外部联结，整个篇章范围内词汇（或短语）之间的关联
    * 指代一般包括回指和共指，指代消歧是衔接性研究的关键问题
* ［篇章连贯性］句子之间的语义关联，又称内部联结，研究主要在信息性和意图性两方面
    * ［篇章信息性］对于接受者，篇章提供的信息超过或低于期望值的程度
    * ［篇章意图性］篇章结构理论不应只考虑篇章内容，还应解释其中意图
    * 汉语重语义，篇章语义主导，衔接性重于连贯性；英语重结构，篇章结构主导，连贯性重于衔接性
* 汉语篇章分析具有“句群”这个本土特征

## 第11章 统计机器翻译
* ［机器翻译方法］基于规则的转换翻译、基于中间语言的翻译方法、基于语料库的翻译方法（包括基于记忆的、基于实例的、基于统计的、基于神经网络的）
* ［噪声信道模型］翻译系统被看作是噪声信道，对一个观察到的信道输出字串S，寻找最大可能的信道输入句子T，即求解T使$$P(T\vert S)$$最大
    * 对于翻译模型$$P(T\vert S)$$的计算，核心使定义目标语言句子T的词与源语言句子S的词之间对应关系，要求“对齐”
    * 通过不同的假设，构建可计算的模型（以词为单位建模的IBM模型）
    * 对位模型是系统的核心，可以利用HMM建模优化
* ［基于短语的模型］如果双语句子中的某些单词序列具有相同的语义，形成的短语具有相同的短语类，那么认为对应的单词序列是对齐的短语
    * 这类模型是最为成熟的统计机器翻译技术 
* …… 

## 第12章 语音翻译
* ［语音翻译核心模块］主要由自动语音识别器（ASR）、机器翻译引擎（MT）和语音合成器（TTS）串行顺序连接组成
* ……

## 第13章 文本分类与情感分类
* ［文本自动分类系统］主要有两种类型，基于知识工程（KE）的分类系统和基于机器学习（ML）的分类系统
* ［文本表示］通常采用向量空间模型（VSM），即把文本转化成特定特征的多维度向量
    * 相似性度量多用内积即余弦值表示 
    * 在VSM表示文前，需要进行词汇化处理，汉语主要依赖分词技术，可用语言无关性的n元语法简化处理
    * 除向量空间模型外，还有词组表示法、概念表示法（WordNet与HowNet）等
* ［文本特征］常用的特征选取方法包括基于文档频率的特征提取法、信息增益法、卡方统计量法和互信息法等
    * 核心是依据一定标准“筛选”特征，降低输入空间的维度，同时要尽量减少噪声
* ［特征权重］衡量某个特征的重要程度或区分能力的强弱，通常利用文本的统计信息 
* ［分类器设计］常用的分类算法包括：朴素贝叶斯、SVM、kNN、ANN、决策树、模糊分类、Roccino分类、Boosting等
* ［分类性能指标］召回率、正确率、F-测度值、微平均和宏平均、平衡点、11点平均正确率
* ［情感分类］较一般的分类问题，具有情感信息表达的隐蔽性、多义性和极性不明显等特征

## 第14章 信息检索与问答系统
* ［信息检索］起源于图书馆的资料查询和文摘索引工作，关键技术包括标引（indexing）和相关度（relevance）计算
    * 估计用户查询标引和候选查询文本之间相关度的模型通常包括：布尔模型、向量空间模型、概率模型和语言模型
    * 大多数信息检索系统都建立主要数据的倒排索引实现快速检索
* ［隐含语义标引模型］（LSI）建立查询文字与文档之间的语义概念关联
    * 采用SVC对词项－文档关联矩阵分解
    * 通过EM迭代，估计PLSI中条件概率等，引入隐含“话题”
    * SPLSI基于PLSI，先展开预聚类
* ［信息检索评测指标］主要包括准确率、召回率、F测度值、P@10、R－precision和最差x％
* ［回答系统］能够接受用户以自然语言形式描述的提问，并从大量异构数据中查找或推断出用户问题答案的信息检索系统
    * 可划分为基于固定语料库的问答系统、网络问答系统和单文本问答系统 
    * 自动回答系统通常由提问处理模块、检索模块和答案抽取模块三部分
    * 问答技术大致分为四种类型：基于检索的问答技术、基于模式匹配的问答技术、基于自然语言理解的问答技术和基于统计翻译模型的问答技术

## 第15章 自动文摘与信息抽取
* ［自动文摘］利用计算机自动实现文本分析、内容归纳合摘要自动生成的技术
* ［多文档摘要］将同一主题下多个文本描述的主要信息按压缩比提炼出一个文本
    * 通常采用基于抽取的方法和基于理解的方法
    * 缺乏自动评测方法，通常关注召回、准确、冗余、偏差
* ［信息抽取］自动抽取指定类型的实体、关系、事件等事实信息，并形成结构化数据输出

## 第16章 口语信息处理与人机对话系统
* ［汉语口语］词长分布集中于1字词和2字词，会出现省略、独词句等非规范语言现象
* 情感表达的主要词类包括形容词、动词和名词
* ［口语解析器］语音翻译系统和人机对话系统中的核心模块
    * ［中间表示格式］特定的信息表示方式，如IF格式通常包括说话者、话语行为、概念序列和参数－属性值对
    * ［基于规则和HMM的统计解析方法］包括词汇分类、语义组块分析、统计解析、语义组块解释、IF生成数个模块
    * ［基于语义决策树］包括训练模块和解析模块；前者生成语义分类树和统计模型，后者利用上述模型对输入句子解析获得领域行为
* ［口语生成方法］采用基于模版的方法和基于特征的深层生成方法相结合的混合生成方法，包括微观规划器、表层生成器和后处理模块
* 对话管理模块是系统的核心，基于对话历史调度人机交互机制，辅助语言解析器对语音识别结果进行正确的理解，为问题求解提供帮助，并指导语言的生成过程