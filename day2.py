#! C:/Users/Admin/.conda/envs/myenv/python.exe
import sys
import os 
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

#结巴分词全模式
sent='在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根节点出发深度探索解空间树。'
wordlist=jieba.cut(sent,cut_all=True)
print"|".join(wordlist)

