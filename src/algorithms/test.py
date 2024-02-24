import numpy as np
import pandas as pd
class TwoDimensionQTable:
    def __init__(self, states,actions,rewords,learningRate=0.1, rewarddecay=0.9, eGreedy=0.9,episode=100):
        #状态列表
        self.states=states
        #动作列表
        self.actions = actions
        #奖励列表
        self.rewords=rewords
        #学习率
        self.lr = learningRate
        #奖励衰减率
        self.gamma = rewarddecay
        #贪婪策略
        self.epsilon = eGreedy
        #训练次数
        self.episode=episode
        #空Q表
        self.qTable = pd.DataFrame(columns=self.actions, dtype=np.float64)
    #遇到新状态后添加到Q表，行索引：状态；列索引：动作
    def stateExist(self, state):
        if str(state) not in self.qTable.index:
            self.qTable = self.qTable.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.qTable.columns,
                    name=state,
                    )
            )
    #根据状态从qTable选择动作
    def chooseAction(self,state):
        #判断状态是否存在Q表中
        self.stateExist(state)
        #选择采取的动作
        #90%的概率按照Q表最优进行选择
        if np.random.uniform()<self.epsilon:
            stateActionList=self.qTable.loc[state,:]
            #若存在多个最好动作，则随机选择其中一个
            action=np.random.choice(stateActionList[stateActionList==np.max(stateActionList)].index)
        #10%的概率随机选择一个动作
        else:
           action=np.random.choice(self.actions)
        return action
    #根据状态和动作返回下一个状态和当前的奖励
    def feedBack(self,state,action):
        nextState=action
        reword=self.rewords.loc[state][nextState]
        return nextState,reword
    #根据当前状态、动作、下一状态、奖励更新Q表
    def updateTable(self,state,action,nextState,reword):
        #判断状态是否存在Q表中
        self.stateExist(nextState)
        #当前状态值
        qPredict=self.qTable.loc[state,action]
        #print("qPredict:",qPredict)
        #下一状态值
        if nextState=='G':
            qTarget=reword
        else:
            qTarget=reword+self.gamma*self.qTable.loc[nextState,:].max()
        #print("qTarget:",qTarget)
        #更新
        self.qTable.loc[state,action]+=self.lr*(qTarget-qPredict)
    #主循环
    def mainCycle(self):
        for episode in range(self.episode):
            #初始状态
            state='A'
            actionStr=""
            while True:
                #选择动作
                action=self.chooseAction(state)
                if episode==self.episode-1:
                    actionStr+=str(state)+"->"
                #状态反馈
                nextState,reword=self.feedBack(state,action)
                #print("nextState:",nextState,"  ","reword:",reword)
                #更新Q表
                self.updateTable(state,action,nextState,reword)
                #更新状态
                state=nextState
                #终止条件
                if state=='G':
                    if episode==self.episode-1:
                        print(actionStr)
                    break
                #break

if __name__=="__main__":
    states=['A','B','C','D','E','F','G']
    rewords=[
        [-99,4,6,6,-99,-99,-99],
        [-99,-99,1,-99,7,-99,-99],
        [-99,-99,-99,-99,6,4,-99],
        [-99,-99,2,-99,-99,5,-99],
        [-99,-99,-99,-99,-99,-99,6],
        [-99,-99,-99,-99,1,-99,8],
        [-99,-99,-99,-99,-99,-99,-99],
        ]
    rewords=pd.DataFrame(rewords,columns=states,index=states)
    actions=['A','B','C','D','E','F','G']
    RL = TwoDimensionQTable(states=states,actions=actions,rewords=rewords)
    RL.mainCycle()
    print(RL.qTable)
