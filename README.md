# gplearn_stock
## 代码作用介绍 
 改进gplearn，主要使用在股票数据挖掘;
## 简介
 本代码在gplearn的基础上，对代码进行调整修改，以期更适用国内的股票策略;
 
 Gplearn是python内部最成熟的符号回归算法实现，作为一种监督学习方法，符号回归试图发现某种隐藏的数学公式，从而利用特征变量预测目标变量;
 
 
 符号回归的具体实现方式是遗传算法。首先生成多个未经历选择公式，此后的每一代中，最合适的公式将被替换;
 
 
 随着伴随次数的增长，它们不断的繁殖，变异，进化，从而不断的逼近数据分布的真相;
 
 
 作为使用到国内二级市场的核心点在在于适应的计算方法，不同适应度的计算方法，得到不同的结果。
 
 
 本篇文章，主要使用调仓收益的预期作为适应度计算公式，对所有的随机生成公式进行进化。
 

## 使用方法：
 安装gplean,将相应的部分进行替换。并结合demo 进行尝试
## 本代码适应度计算方法：
 按照公式的大小提取前50（或100）只股票，等比例建仓，按照5个交易日进行调仓，适应度为股票收益率的均值，越大适应度越高，不设上限。
 具体方法请查看_program.py的stock_excute函数，如需其他适应度，也可对此部分进行调整修改。
