import os
import random
import pygame
import numpy as np
import collections
from time import sleep
from datetime import datetime
from sklearn.neural_network import MLPRegressor

GAMES = 5
TABLE_HEIGHT = 5
TABLE_WIDTH = 5

INPUTS = TABLE_HEIGHT * TABLE_WIDTH + 6
TREE_SEARCHES = 100
TREE_DEPTH = 10
DISCOUNT = 0.5
EPSILON = 0.5

TRAIN = False
SAVE_PROGRESS = False
SHOW_TURNS = True
SHOW_INFO = False






score = 0
connectionsList = []
net = MLPRegressor(hidden_layer_sizes=(10,))
net.fit([[0] * INPUTS], [0])

class Visual():
    def __init__(self):
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()
        self.screenWidth = 500
        self.screenHeight = 700
        self.windowSize = (self.screenWidth, self.screenHeight)
        self.screen = pygame.display.set_mode(self.windowSize)
        pygame.display.set_caption("Neural Network Game")

        self.tileWidth = 400 / max(TABLE_HEIGHT, TABLE_WIDTH)
        self.tileHeight = self.tileWidth
        self.gapSize = self.tileWidth * 0.1

        self.centerPoint = []
        for demension in (TABLE_WIDTH, TABLE_HEIGHT):
            a = range(demension)
            while len(a) > 2:
                a = a[1:-1]

            if len(a) == 2:
                self.centerPoint.append((a[0] + a[1]) / 2)
            else:
                self.centerPoint.append(a[0])
        self.centerPoint = tuple(self.centerPoint)

        self.textures = [pygame.image.load('textures/{}.png'.format(i)) for i in range(1, 9)]

    def render_board(self, board, score, turn):
        self.screen.fill((242, 239, 235))

        for row in range(TABLE_HEIGHT):
            for column in range(TABLE_WIDTH):
                tile_value = board[row][column]

                if int(tile_value - 1) >= len(self.textures):
                    self.textures.append(pygame.image.load('textures/{}.png'.format(int(tile_value))))
                texture = pygame.transform.scale(self.textures[int(tile_value - 1)], (int(self.tileWidth), int(self.tileHeight)))

                self.screen.blit(texture, (self.screenWidth / 2 - self.tileWidth / 2 - (self.tileWidth + self.gapSize) * (self.centerPoint[0] - column), self.screenHeight / 2 - self.tileHeight / 2 - (self.tileHeight + self.gapSize) * (self.centerPoint[1] - row)))

        myfont = pygame.font.SysFont('Arial', 70)
        heightMultiplier = 0.02
        for text in ('Score {}'.format(score), 'Turn {}'.format(turn)):
            textsurface = myfont.render(text, False, (0, 0, 0))
            size = myfont.size(text)
            self.screen.blit(textsurface, (self.screenWidth / 2 - size[0] / 2, self.screenHeight * heightMultiplier))
            heightMultiplier = 0.1

    def update_screen(self):
        pygame.display.update()
        pygame.event.get()

def convertToTuple(arr):
    tupArr = [tuple(elem) for elem in arr]
    return tuple(tupArr)

def isLost(state):
    table = state[0]
    results = [(table[int(i[1][0] / TABLE_WIDTH)][i[1][0] % TABLE_WIDTH], table[int(i[1][1] / TABLE_WIDTH)][i[1][1] % TABLE_WIDTH]) for i in connectionsList]

    for i in results:
        if abs(i[0] - i[1]) == 1:
            return False
    return True

def getActions(state):
    actions = []

    table = state[0]

    for indx, value in enumerate([(table[int(i[1][0] / TABLE_WIDTH)][i[1][0] % TABLE_WIDTH], table[int(i[1][1] / TABLE_WIDTH)][i[1][1] % TABLE_WIDTH]) for i in connectionsList]):
        if abs(value[0] - value[1]) == 1:
            bit = 0
            if value[1] > value[0]:
                bit = 1

            nums = (connectionsList[indx][1][1 - bit], connectionsList[indx][1][bit])
            actions.append(((int(nums[0] / TABLE_WIDTH), nums[0] % TABLE_WIDTH), (int(nums[1] / TABLE_WIDTH), nums[1] % TABLE_WIDTH)))

    return actions

def generateTable(table):
    for i in range(TABLE_HEIGHT):
        for j in range(TABLE_WIDTH):
            table[i, j] = random.randrange(2) + 1
    return table

def turnsAvailible(table):
    result = 0
    for i in [(table[int(i[1][0] / TABLE_WIDTH)][i[1][0] % TABLE_WIDTH], table[int(i[1][1] / TABLE_WIDTH)][i[1][1] % TABLE_WIDTH]) for i in connectionsList]:
        if abs(i[0] - i[1]) == 1:
            result += 1
    return result

def turnExists(table):
    for i in [(table[int(i[1][0] / TABLE_WIDTH)][i[1][0] % TABLE_WIDTH], table[int(i[1][1] / TABLE_WIDTH)][i[1][1] % TABLE_WIDTH]) for i in connectionsList]:
        if abs(i[0] - i[1]) == 1:
            return True
    return False

def makeTurn(state, action):
    global score
    scored = False

    table = np.array(state[0])

    table[action[0][0], action[0][1]] = 0
    table[action[1][0], action[1][1]] += 1


    turnScore = table[action[1][0], action[1][1]] ** 2

    if table[action[1][0]][action[1][1]] > score:
        scored = True
        turnScore = 10000

    if action[0][0] > 0:
        for j in range(action[0][0], 0, -1):
            table[j, action[0][1]] = table[j - 1, action[0][1]]
            table[j - 1, action[0][1]] = 0
    table[0, action[0][1]] = (random.randrange(5) + 1) % 2 + 1

    if scored == False:
        multiplier = 1
        closeNumbers = [(action[1][0], action[1][1])]

        while closeNumbers != []:
            differnce = 1
            newNumbers = []
            for number in closeNumbers:
                sidesVertical = []
                sidesHorizontal = []

                if number[0] > 0:
                    sidesVertical.append(-1)
                if number[0] < TABLE_HEIGHT - 1:
                    sidesVertical.append(1)
                if number[1] > 0:
                    sidesHorizontal.append(-1)
                if number[1] < TABLE_WIDTH - 1:
                    sidesHorizontal.append(1)

                for side in sidesVertical:
                    if table[number[0] + side, number[1]] == score:
                        multiplier += 1
                    if table[number[0] + side, number[1]] - table[number[0], number[1]] == differnce:
                        newNumbers.append((number[0] + side, number[1]))

                for side in sidesHorizontal:
                    if table[number[0], number[1] + side] == score:
                        multiplier += 1
                    if table[number[0], number[1] + side] - table[number[0], number[1]] == differnce:
                        newNumbers.append((number[0], number[1] + side))
            multiplier += len(newNumbers)
            closeNumbers = newNumbers
            differnce = 2

        turnScore = turnScore ** multiplier

    return table, turnScore, scored

def searchTree(currState, action):
    global TREE_DEPTH

    prevState = currState
    depth = 0
    utility = 0

    while ((not isLost(prevState)) and depth < TREE_DEPTH):
        if depth != 0:
            action = random.choice(getActions(prevState))
        table, reward, _ = makeTurn(prevState, action)

        newState = (convertToTuple(table), prevState[1]-1)
        prevState = newState
        utility += reward * (DISCOUNT**depth)
        depth += 1
    return utility

def getTable(state):
    result = []
    for i in state[0]:
        for j in i:
            result.append(j)
    return result

def getNetData(state, action):
    minimalRow = min(action[0][0], action[1][0])
    maximumRow = max(action[0][0], action[1][0])
    isSameColumn = 1 if action[0][1] == action[1][1] else 0
    numberOfTurns = turnsAvailible(state[0])

    maximumPossibleNumber = 0
    table = state[0]
    if table[action[1][0]][action[1][1]] - table[action[0][0]][action[0][1]] == 1:
        maximumPossibleNumber = table[action[1][0]][action[1][1]] + 1
    averageReward = np.median([searchTree(state, action) for i in range(TREE_SEARCHES)])

    phi = [minimalRow, maximumRow, isSameColumn, numberOfTurns, maximumPossibleNumber, averageReward]
    phi += getTable(state)

    if SHOW_INFO:
        print(phi)

    return np.array(phi)

def getNetOutput(state, action):
    global net

    if isLost(state):
        return 0
    return net.predict([getNetData(state, action)])[0]

def updateWeights(state, action, reward, newState):
    global net
    a = [(getNetOutput(newState, action), action) for action in getActions(newState)]
    if a != []:
        multiplier, _ = max(a)

        net.partial_fit([getNetData(state, action)], [reward + DISCOUNT * multiplier])

def playGame(table, sample, visual = 0):
    global score
    score = 2
    turn = 0
    #prevWeights = np.zeros(31, dtype = float)

    if not TRAIN:
        visual.render_board(table, score, turn)
        visual.update_screen()

    while turnExists(table):
        startTime = datetime.now()

        if SHOW_TURNS:
            print('')
            print('Turn:', turn)
            print('SCORE', score)
            print(table)

        currState = (convertToTuple(table), turn)

        action = None
        if (random.random() < EPSILON):
            if SHOW_INFO:
                print('Random turn')
            action = random.choice(getActions(currState))
        else:
            if SHOW_INFO:
                print('Neural network turn')
            _, optimalAction = max((getNetOutput(currState, action), action) for action in getActions(currState))
            action = optimalAction
        if SHOW_TURNS:
            print('Making turn from', action[0], 'to', action[1])
        table, turnScore, scored = makeTurn(currState, action)
        if scored == True:
            score += 1
        turn += 1
        newState = (convertToTuple(table), turn)
        
        if not TRAIN:
            while (datetime.now() - startTime).total_seconds() < 2:
                sleep(0.1)

            visual.render_board(table, score, turn)
            visual.update_screen()

        if TRAIN:
            if SHOW_INFO:
                print('Updating neural network weights')
            updateWeights(currState, action, turnScore, newState)

        #if SHOW_INFO:
        #    print('new weights (scaled):')
        #    print(net.coefs_, net.intercepts_)
    if SHOW_TURNS:
        print('')
        print('Turn:', turn)
        print(table)
    print(sample, 'FINAL SCORE', score)
    return score

def main():
    global connectionsList, score
    random.seed()
    np.random.seed()

    scoreSum = 0
    score = 2

    horizontalPairs = [(i, i + 1) for i in range(TABLE_WIDTH * TABLE_HEIGHT) if i % TABLE_WIDTH != TABLE_WIDTH - 1]
    verticalPairs = [(i, i + TABLE_WIDTH) for i in range(TABLE_WIDTH * TABLE_HEIGHT) if int(i / TABLE_WIDTH) != TABLE_HEIGHT - 1]
    pairsList = sorted(horizontalPairs + verticalPairs)
    connectionsList = [(i, (pairsList[i][0], pairsList[i][1])) for i in range((TABLE_HEIGHT - 1) * TABLE_WIDTH + (TABLE_WIDTH - 1) * TABLE_HEIGHT)]


    if not TRAIN:
        global EPSILON

        visual = Visual()

        EPSILON = 0

        net.coefs_ = np.load('coefs.npy', allow_pickle=True)
        net.intercepts_ = np.load('intercepts.npy', allow_pickle=True)

        net.n_layers_ = 3
        net.n_outputs_ = 1
        net.out_activation_ = "identity"

    for i in range(GAMES):
        table = np.zeros((TABLE_HEIGHT, TABLE_WIDTH), dtype = int)
        table = generateTable(table)
        if TRAIN:
            score = playGame(table, i+1)
        else:
            score = playGame(table, i+1, visual)
        scoreSum += score
        print(i+1, 'AVERAGE SCORE:', float(scoreSum)/(i+1))
        print('')

    if TRAIN:
        print('new weights (scaled):')
        print(net.coefs_, net.intercepts_)

        if SAVE_PROGRESS:
            np.save('coefs.npy', net.coefs_)
            np.save('intercepts.npy', net.intercepts_)

if __name__ == "__main__":
    main()