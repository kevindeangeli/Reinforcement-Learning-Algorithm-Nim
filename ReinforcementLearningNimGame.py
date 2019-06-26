"""
Created by Kevin De Angeli
01/21/19
This code was partially inspired by Wesley Tansey's program which can be found in:
https://github.com/tansey/rl-tictactoe/blob/master/tictactoe.py
Wesley Tansey's program is based on the reference implementation of the Tic-Tac-Toe value
function learning agent described in Chapter 1 of
"Reinforcement Learning: An Introduction" by Sutton and Barto.
"""


import itertools
import random
from copy import copy, deepcopy
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np





state =[]
EMPTY=0

def getNewGameChips():
    return [3,5,7]

def get_board(state):
    # Get initial board
    print("*" * 25)
    print("Board:")
    print("*" * 25)
    for i,x in enumerate(state):
        print('Pile {}: {}'.format(i + 1, 'O' * state[i]), "   #=", state[i] )
    print("-" * 25)

def gameover(state, player):
    for i, x in enumerate(state):
        if x != 0:
            return 0
    return player



class Agent(object):
    def __init__(self, player, lossval = 0, learning = True): #Player is called with either 0or1
        self.values = {} #This is a list of different scenarios encounterd, associated with a reward in each scenario.
        self.player = player
        self.lossval = lossval
        self.learning = learning #If this is false, then the algorithm stops learning from experience (backup method)
        self.epsilon = 0.20       #Probability that the algorithm will "explore" vs "take the greedy approach"
        self.alpha = 0.99        #temporal-dierence learning method. Affects  V(s') - V(s) == NewState-PreviousState
        self.prevstate = None    #Used in the Backup function to learn from experiecne
        self.prevscore = 0       #Used in the Backup function to learn from experiecne
        self.count = 0

    def moveExecute(self, state, move):
        newState = state.copy()
        pile = move[0]
        previousChips = newState[pile - 1]
        newChips = previousChips - move[1]
        newState[move[0] - 1] = newChips
        return newState

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

#######################################################################
#The Agent hast two actions: Random with probability "epsilon"
                        #and  Greedy with probability 1 - "epsilon"
#######################################################################
    def action(self, state):
        r = random.random()  #Returns a number in (0,1) This random is from the Library Import.
        if r < self.epsilon:  #Note: when Epsilon is 0, this never happens.(i.e. Always Greedy)
            move = self.randomMove(state) 
            print("Random")
        else:
            move = self.greedy(state)
            print("Greedy")
        self.prevstate = tuple(self.moveExecute(state=state,move=move))
        self.prevscore = self.lookup(tuple(self.moveExecute(state,move)))
        return move
# **********************************************************************



#######################################################################
#Random and Greedy are the two possible actions an Agent can take.
#######################################################################
    def randomMove(self, state):
        available = self.generatePossibleMoves(state)
        #print("available Moves: ", available)
        return list(random.choice(list(available))) #Random.Choice picks one from the list of pairs.

#The values associated with each move change depending on the number of times the computer
#play against itself.
    def greedy(self, state):
        maxval = -50000 #This is an unnecessarily small number. It just need to be less than -1.
        maxmove = None  
        copyState=state.copy()
        total_moves = self.generatePossibleMoves(copyState)
        for i,x in enumerate(total_moves):
            copyState = state.copy()
            do_move = self.moveExecute(state=copyState,move=x)
            val= self.lookup(do_move)
            if val > maxval:
                maxval=val
                maxmove=x
        self.backup(maxval)
        return maxmove

#The Backup method:
    #Once you have choisen the best move at at the present state, you can go back and modify the value of the
    #previous state.

    #Think about it this way: You are in one situation and you chose you best move.
    #Knowing what your best move is right now will give you a clue of how good you previous state was to begin with.

    #Consider the possible scenarios:
        #1. the maxval of the previous state is greater than the maxval of this new state.
            #Then: nextval - self.prevscore < 0, and the value of the previous state will decrease.
        #2. The maxval of the previous state is the same than the maxval of this new state.
            #Then, nextval - self.prevscore = 0. And the maxval of the previous state remains the same.
        #3. The maxval of this new state is greater than the maxval of the previous state.
            #Then, nextval - self.prevscore > 0 and this make the previous state valuable:
            #That is because being in the previous state will potentially allow you to be in this current state
            #in which the value is greater.

    def backup(self, nextval): 
        #print("-------------------------")
        #print("prev state", self.prevstate)
        if self.prevstate != None and self.learning: #You start to see more numbers non-equal to .5
            #print("value Before=", self.values[self.prevstate])
            #print("Nextval ", nextval)
            #print("prevscore=", self.prevscore)
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            #print("prevstate=",self.prevstate)
            #print("Newvalue", self.values[self.prevstate])
            #print("values total", self.values)

# **********************************************************************


#######################################################################
# The Functions Lookup, Add, and winnerval work together:
# 1. Lookup: It checks if this scenario was already encountered, and
    # returns the value associated with this scenario.
# 2. If a previous scenario is not found, it calls the add function.
# 3. The Add function calls the Winnerval function which associates
    # a value with the scenario.
# This value depends on the winner (i.e. P1, P2, Draw, no winner yet)
#######################################################################

    def lookup(self,state):
        key = tuple((state))
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self,state):
        winner = gameover(state=state,player=self.player)
        tup = tuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        else:
            return self.lossval
#**********************************************************************


    # This function generates all the possible moves given
    # They are produced in the form: (i,j), where i is the pile, j is the number of chips to take.
    def generatePossibleMoves(self,state):  # Will this function go into the Agent Class?
        q = []
        for i, x in enumerate(state):
            j = 1
            while j <= x:
                b = (i + 1, j)
                q.append(b)
                j = j + 1
        return q

    def newGameReset(self, winner):
        if winner == self.player:
            self.values[self.prevstate] = 1
        else:
            self.values[self.prevstate] = 0
        self.prevstate = None
        self.prevscore = 0




#**********************************************************************
#***********************END OF CLASS **********************************
#**********************************************************************
def new_nim_sum(listOfRocks, randPile):

    nim = 0

    # Calculate nim sum for all elements in the listOfRocks
    for i in listOfRocks:
        nim = nim ^ i

    print("Hint: nim sum is {}.".format(nim))

    # Determine how many rocks to remove from which pile
    stones_to_remove = max(listOfRocks) - nim
    stones_to_remove = abs(stones_to_remove)

    # Logic for certain configurations on determining how many stones to remove from which pile
    # "listOfRocks.index(max(listOfRocks))+ 1 )" determines the index in listOfRocks at which the biggest
    # pile of stones exists.
    if (nim > 0) and (len(listOfRocks) > 2) and (nim != max(listOfRocks)) and (nim != 1):
        print("Pick {} stones from pile {}".format(stones_to_remove, listOfRocks.index(max(listOfRocks)) + 1))

    if (nim > 0) and (len(listOfRocks) > 2) and (nim == max(listOfRocks)) and (nim != 1):
        print("Pick {} stones from pile {}.".format(nim, listOfRocks.index(max(listOfRocks)) + 1))

    if nim > 0 and len(listOfRocks) <= 2 and (stones_to_remove != 0):
        print("Pick {} stones from pile {}".format(stones_to_remove, listOfRocks.index(max(listOfRocks)) + 1))

    if nim > 0 and len(listOfRocks) <= 2 and (stones_to_remove == 0):
        print("Pick {} stones from pile {}".format(nim, listOfRocks.index(max(listOfRocks)) + 1))

    elif (nim == 1) and (len(listOfRocks) <= 2):
        print("Pick {} stones from pile {}".format(nim, listOfRocks.index(max(listOfRocks)) + 1))

    if (nim == 1) and (nim == max(listOfRocks)) and (nim != 0) and (len(listOfRocks) > 2):
        print("Pick {} stones from pile {}".format(nim, listOfRocks.index(max(listOfRocks)) + 1))

    if nim == 0:
        print("Pick all stones from pile {}.".format(listOfRocks.index(max(listOfRocks)) + 1))

def new_new_nim_sum():
    state = [3, 5, 7]
    nim = 0
    SumZeroNimStages = []

    for i, x in enumerate(total_game_scenarios):
        nim = 0
        listOfRocks = x
        # Calculate nim sum for all elements in the listOfRocks
        for i in listOfRocks:
            nim = nim ^ i

            # Determine how many rocks to remove from which pile
        stones_to_remove = max(listOfRocks) - nim
        stones_to_remove = abs(stones_to_remove)

        if nim == 0 and listOfRocks not in SumZeroNimStages:
            SumZeroNimStages.append(listOfRocks)

    # Note that there are 4*6*8=192 possible scenarios.
    # print("Sum Zero Stages: ", SumZeroNimStages)
    # print("Number of Zero sum Stages = ", len(SumZeroNimStages))
    return SumZeroNimStages


def plotData(y1,years):


    xdata1 = list(range(0,years))
    figure(figsize=(10, 7))

    #plt.subplot(2, 1, 1)
    #plt.axis(ymin = 0, ymax=24)
    plt.plot(xdata1, y1)
    plt.title('Reinforcement Learning - β = 0.50')
    plt.ylabel('Nim zero sum (Max = 24)')
    plt.yticks(np.arange(0, 25, step=2))
    #plt.xticks(np.arange(0, years, step=years/10))
    plt.xlabel('Number of games played')
    #plt.grid(True)

    #plt.savefig('Graph#.png')
    plt.show()



def analyzeResults(agent):
    nimSumZero = []
    totalNumCoincidencesWithNimSumZero = 0
    nim_states = new_new_nim_sum()
    temp_value = 0
    check = False
    for i, x in enumerate(nim_states):
        if tuple(x) in agent.values:
            temp_value = agent.values[tuple(x)]
        if temp_value > 0.9:
            totalNumCoincidencesWithNimSumZero += 1
            nimSumZero.append(x)
            temp_value = 0

    print(nimSumZero)
    return totalNumCoincidencesWithNimSumZero





def play(agent1, agent2):
    state = getNewGameChips()
    i=random.randint(0,9)
    while sum(state) != 0:
        print("in Play State: ",state)
        new_nim_sum(listOfRocks=state, randPile=len(state))
        if i % 2 == 0: #Player 1 plays when i=0
            print("Player 1 plays:")
            move = agent1.action(state)
            print("move:", move)
            state= agent1.moveExecute(state=state,move=move)
            winner = gameover(state, agent1.player)
        else: #Player 2 plays when i=1
            print("Player 2 plays:")
            move = agent2.action(state)
            print("move:", move)
            state = agent2.moveExecute(state=state, move=move)
            winner = gameover(state, agent2.player)
        print("New state:", state)
        print("*************************")
        i = i+1
        if winner != EMPTY:
            agent1.newGameReset(winner)
            agent2.newGameReset(winner)
            return winner
    return winner

def generateAllScenarios():
    scenarios = []
    for i in range(4):
        for k in range(6):
            for q in range(8):
                scenarios.append([i, k, q])
    return scenarios
P1=Agent(player=1, learning=True)
P2=Agent(player=2, learning=True)

total_game_scenarios = generateAllScenarios()

dataForPlottingNimSumZero = []

numberOfGames_and_years = 2500
for i in range(numberOfGames_and_years):
    print(play(P1,P2))
    #print("P1 ", P1.values)
    #print("P2 ", P2.values)
    print("----------End of Game ", i, "--------")
    dataForPlottingNimSumZero.append(analyzeResults(P1))
plotData(dataForPlottingNimSumZero, years=numberOfGames_and_years)
print("Analize", dataForPlottingNimSumZero)

P1.epsilon=0
P2.epsilon=0
#for i in range(10):
    #print(play(P1,P2))
    #print("----------End of Game ", i, "--------")



print("P1 Values", P1.values)
print("P2 Values", P2.values)

print("P1 values len", len(P1.values)) 
