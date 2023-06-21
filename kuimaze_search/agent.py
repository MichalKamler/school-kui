import time, kuimaze, os, random
from queue import PriorityQueue

def heuristicFun(cell1, cell2): #returns heroistic len from cell to other cell(goal)
    x1, y1 = cell1
    x2, y2 = cell2
    return float(abs(x1 - x2) + abs(y1 - y2))

class Agent(kuimaze.BaseAgent):
    def __init__(self, environment):
        self.environment = environment

    def find_path(self):
        '''
        Method that must be implemented by you. 
        Expects to return a path as a list of positions [(x1, y1), (x2, y2), ... ].
        '''
        observation = self.environment.reset() # must be called first, it is necessary for maze initialization
        goal = observation[1][0:2]                                   # my goal (x, y)
        position = observation[0][0:2]                               # initial state (x, y)
        if(goal==position):     # resolving trivial case
            return [goal]
        gScore, fScore = {}, {}
        gScore[position] = float(0)     #dic for positions and their gscore(len of way from start)
        fScore[position] = heuristicFun(position, goal)     #dic of positons and their heuristic len to goal
        openTiles = PriorityQueue()
        openTiles.put((heuristicFun(position, goal) + 0, heuristicFun(position, goal), position)) #fcost, hcost, cell
        pathDic, iteratedTiles = {}, []

        while not openTiles.empty(): #starting astar
            position = openTiles.get()[2]       #cell i am testing from queue
            if(position in iteratedTiles):      #expanded to tile I already checked
                continue
            iteratedTiles.append(position)
            if(position==goal):     #Astar complete
                break    
            new_positions = self.environment.expand(position)         # [[(x1, y1), cost], [(x2, y2), cost], ... ]
            for i in range(len(new_positions)):     #loop for going through all my newPositions and possibly updating temps in my dic
                tempG = gScore[position] + new_positions[i][1]
                tempF = tempG + heuristicFun(new_positions[i][0], goal)
                if(new_positions[i][0] not in gScore):#if I have not been in this tile I update my dic with temp gscore and temp fscore
                    gScore[new_positions[i][0]] = tempG
                if(new_positions[i][0] not in fScore):
                    fScore[new_positions[i][0]] = tempF
                if(tempF <= fScore[new_positions[i][0]]):
                    gScore[new_positions[i][0]] = tempG
                    fScore[new_positions[i][0]] = tempF
                    openTiles.put((tempF, heuristicFun(new_positions[i][0], goal),  new_positions[i][0]))
                    pathDic[new_positions[i][0]] = position                                 
            #    self.environment.render()               # show enviroment's GUI       DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION!      
            #    time.sleep(0.1)                         # sleep for demonstartion     DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION! 


        if(goal not in pathDic): #if it is not possible to reach goal i return none
            return None
        path = [goal]        # create path as list of tuples in format: [(x1, y1), (x2, y2), ... ]  
        while goal!=observation[0][0:2]: #i will be recreating path from goal to start and changing the value of goal in the process
            goal = pathDic[goal] 
            path.append(goal)
        path.reverse() # i have to reverse path, because it is from goal to start
        return path


if __name__ == '__main__':

    MAP = 'maps_difficult/maze100x100.png'
    #MAP = 'maps/normal/normal12.bmp'
    #MAP = 'maps/easy/easy1.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path = agent.find_path()
    print(path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(20)
