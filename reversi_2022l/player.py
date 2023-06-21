import copy
import math
import time
#python .\reversi_creator.py player novak vojta playerSortMove playerRandom playerNoDeepCopy novakAB playerNewSort playerTime playerOldcopy

class MyPlayer():
    '''Alpha-beta pruning with changing heuristic board and adaptive depth''' # TODO a short description of your player
    def __init__(self, my_color,opponent_color, board_size=8): #declares colors, heuristic
        self.name = 'B0t1n_b0T_F0r_B0tN3$$_V2.0' #TODO: fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size
        self.empty_color = -1
        self.howManyTimesRecursive = 0
        self.maxTimerecursive = 0
        self.howManyTimesDidNotIncraseDepth = 0
        if(my_color==0):
            self.p1_color = 0
            self.p2_color = 1
        else:
            self.p1_color = 1
            self.p2_color = 0
        if(board_size==6):
            self.depth = 7
            self.stoneVal = [
        [220, -50, 8, 8, -50, 220],
        [-50, -75, 1, 1, -75, -50],
        [8, 1, 2, 2, 1, 8],
        [8, 1, 2, 2, 1, 8],
        [-50, -75, 1, 1, -75, -50],
        [220, -50, 8, 8, -50, 220]
        ]
        if(board_size==8):
            self.depth = 5
            self.stoneVal = [
        [220, -50, 10, 5, 5, 10, -50, 220],
        [-50, -75, 1, 1, 1, 1, -75, -50],
        [10, 1, 3, 2, 2, 3, 1, 10],
        [5, 1, 2, 1, 1, 2, 1, 5],
        [5, 1, 2, 1, 1, 2, 1, 5],
        [10, 1, 3, 2, 2, 3, 1, 10],
        [-50, -75, 1, 1, 1, 1, -75, -50],
        [220, -50, 10, 5, 5, 10, -50, 220],
        ]
        if(board_size==10):
            self.depth = 4
            self.stoneVal = [
        [220, -60, 30, 15, 15, 15, 15, 30, -60, 220],
        [-60, -90, -10, -10, -10, -10, -10, -10, -90, -60],
        [30, -10, 15, 3, 3, 3, 3, 15, -10, 30],
        [15, -10, 3, 3, 3, 3, 3, 3, -10, 15],
        [15, -10, 3, 3, 3, 3, 3, 3, -10, 15],
        [15, -10, 3, 3, 3, 3, 3, 3, -10, 15],
        [15, -10, 3, 3, 3, 3, 3, 3, -10, 15],
        [30, -10, 15, 3, 3, 3, 3, 15, -10, 30],
        [-60, -90, -10, -10, -10, -10, -10, -10, -90, -60],
        [220, -60, 30, 15, 15, 15, 15, 30, -60, 220]
        ]

    def move(self,board):   #returns best move I found
        self.startTime = time.time()
        self.checkCorners(board)
        self.changeEval(board)
        score, ABmove = self.alpha_beta(board, self.depth, -math.inf, math.inf, True)

        if(time.time()-self.startTime>4.9):
            self.depth -= 1 
            self.howManyTimesDidNotIncraseDepth = 0
        else:
            self.howManyTimesDidNotIncraseDepth+=1
        if(self.howManyTimesDidNotIncraseDepth>=10):
            self.depth +=1

        #self.maxTimerecursive = max(self.maxTimerecursive, self.howManyTimesRecursive)
        #print(self.howManyTimesRecursive, self.maxTimerecursive)
        self.howManyTimesRecursive = 0
        return ABmove
    
    def changeEval(self, board):    #sometime in game changes my heuristic board
        sum = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if(board[i][j]==-1):
                    sum += 1
        if(sum/(len(board)*len(board))<=0.2):
            for i in range(len(board)):
                for j in range(len(board[0])):
                    self.stoneVal[i][j] = 1
            for i in [0, len(board)-1]:
                for j in [0, len(board)-1]:
                    self.stoneVal[i][j] = 10

    def checkCorners(self, board):      #checks if I have corners and if I do it adjusts heuristic board
        if(board[0][0]==self.my_color):
            self.stoneVal[1][0] = 10
            self.stoneVal[1][1] = 10
            self.stoneVal[0][1] = 10
        if(board[0][len(board)-1]==self.my_color):
            self.stoneVal[1][len(board)-1] = 10
            self.stoneVal[1][len(board)-2] = 10
            self.stoneVal[0][len(board)-2] = 10
        if(board[len(board)-1][0]==self.my_color):
            self.stoneVal[len(board)-1][1] = 10
            self.stoneVal[len(board)-2][1] = 10
            self.stoneVal[len(board)-2][0] = 10
        if(board[len(board)-1][len(board)-1]==self.my_color):
            self.stoneVal[len(board)-1][len(board)-2] = 10
            self.stoneVal[len(board)-2][len(board)-2] = 10
            self.stoneVal[len(board)-2][len(board)-1] = 10

    def eval(self, board):  #evaluetes score of the board depeniding on current heuristic board
        score = 0
        score = sum([self.stoneVal[i][j] for i in range(len(board)) for j in range(len(board[i])) if board[i][j] == self.my_color])
        score -= sum([self.stoneVal[i][j] for i in range(len(board)) for j in range(len(board[i])) if board[i][j] == self.opponent_color])
        return score

    def alpha_beta(self, board, depth, alpha, beta, maximizingPlayer): #alpha beta pruning
        if (depth == 0 or self.gameOver(board)):
            return self.eval(board), None
        
        if maximizingPlayer:
            movesList = self.get_all_valid_moves(board, self.my_color)
            if not movesList:
                return self.eval(board), None
            maxEval = -math.inf
            bestMove = None
            if(time.time()-self.startTime<4.9):
                for move in movesList:
                    newBoard = self.updateBoard(board, move, self.my_color)
                    eval, _ = self.alpha_beta(newBoard, depth - 1, alpha, beta, False)
                    if eval > maxEval:
                        maxEval = eval
                        bestMove = move
                    alpha = max(alpha, eval)
                    if alpha >= beta:
                        break
            return maxEval, bestMove
        else:
            movesList = self.get_all_valid_moves(board, self.opponent_color)
            if not movesList:
                return self.eval(board), None
            minEval = math.inf
            bestMove = None
            if(time.time()-self.startTime<4.9):
                for move in movesList:
                    newBoard = self.updateBoard(board, move, self.opponent_color)
                    eval, _ = self.alpha_beta(newBoard, depth - 1, alpha, beta, True)
                    if eval < minEval:
                        minEval = eval
                        bestMove = move
                    beta = min(beta, eval)
                    if alpha >= beta:
                        break
            return minEval, bestMove
    
    def gameOver(self, board): #checks if game is over at some stage of the game
        myMoves = self.get_all_valid_moves(board, self.my_color)
        opMoves = self.get_all_valid_moves(board, self.opponent_color)
        if(myMoves==None and opMoves==None):
            return True
        else:
            return False

    def updateBoard(self, board, move, playercolor): #board, (x,y), 1/0
        newboard = copy.deepcopy(board)
        newboard = self.play_move(move, playercolor, newboard)
        return newboard

#code from here is just s slightly changed(it can play move for both colors) code from the zip file we recieved

    def play_move(self, move, players_color, changedBoard):
        '''
        :param move: position where the move is made [x,y]
        :param player: player that made the move
        '''

        changedBoard[move[0]][move[1]] = players_color
        dx = [-1,-1,-1,0,1,1,1,0]
        dy = [-1,0,1,1,1,0,-1,-1]
        for i in range(len(dx)):
            if self.confirm_direction(move,dx[i],dy[i],players_color,changedBoard):
                self.change_stones_in_direction(move,dx[i],dy[i],players_color,changedBoard)    
        return changedBoard
    
    def confirm_direction(self,move,dx,dy,players_color,changedBoard):
        '''
        Looks into dirextion [dx,dy] to find if the move in this dirrection is correct.
        It means that first stone in the direction is oponents and last stone is players.
        :param move: position where the move is made [x,y]
        :param dx: x direction of the search
        :param dy: y direction of the search
        :param player: player that made the move
        :return: True if move in this direction is correct
        '''
        if players_color == self.p1_color:
            opponents_color = self.p2_color
        else:
            opponents_color = self.p1_color

        posx = move[0]+dx
        posy = move[1]+dy
        if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
            if changedBoard[posx][posy] == opponents_color:
                while (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                    posx += dx
                    posy += dy
                    if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                        if changedBoard[posx][posy] == self.empty_color:
                            return False
                        if changedBoard[posx][posy] == players_color:
                            return True

        return False

    def change_stones_in_direction(self,move,dx,dy,players_color,changedBoard):
        posx = move[0]+dx
        posy = move[1]+dy
        while (not(changedBoard[posx][posy] == players_color)):
            changedBoard[posx][posy] = players_color
            posx += dx
            posy += dy

    def __is_correct_move(self, move, board, color):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board, color)[0]:
                return True, 
        return False

    def __confirm_direction(self, move, dx, dy, board, color):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if(color == 0):
            oppositecolor = 1
        else:
            oppositecolor = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == oppositecolor:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board, color):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board, color):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            #print('No possible move!')
            return []
        return valid_moves
    