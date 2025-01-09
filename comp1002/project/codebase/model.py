from collections import deque
from enum import Enum, auto
import logging
import numpy as np
import secrets as sc

ROTATION={ #rotations
    "u": [1,0],
    "d": [-1, 0],
    "l": [0, -1],
    "r": [0, 1]
}
SPARSITY=2 # 1=most dense

class GameStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    END = auto()

class GameLogic:

    def __init__(self):
        self.__n=0
        self.__m=0
        self.__board = None
        self.__dict=dict()
        self.__observers=[]
        self.__game_status=GameStatus.IDLE

    def new_game(self, width, entity_type):
        self.__log("Creating New Game")
        self.__n = int(width)
        self.__m = int(entity_type)
        self.__dict.clear()
        self.__board=np.zeros((self.__n, self.__n), dtype=int)
        while not self.is_solution():
            self.__dict = {key: [] for key in range(1, self.__m+1)}
            self.__fill_board()
            self.__set_dict()
        self.__game_status=GameStatus.RUNNING
        self.__update_game()

    def __log(self, message: str):
        print(f"[Model]: {message}")

    def subscribe(self, observer):
        self.__observers.append(observer)

    def __update_game(self):
        self.__log("Notifying Observers")
        info = [self.__n, self.__m, self.__board, self.__dict, self.__game_status]
        for observer in self.__observers:
            observer.update(info)

    def __fill_board(self): #generate a random board
        for entity in range(1, self.__m+1):
            for _ in range(sc.randbelow(self.__n**2//(self.__m*SPARSITY)+1)*2):
                row, col = self.__get_valid_pos()
                self.__board[row][col] = entity

    def __set_dict(self):
        for row, row_list in enumerate(self.__board):
            for col, entity in enumerate(row_list):
                if entity:
                    self.__dict[entity].append((row, col))

    def __get_valid_pos(self):
        while True: 
            row, col = sc.randbelow(self.__n), sc.randbelow(self.__n)
            if not self.__board[row][col]:
                return row, col
    
    def print_all(self):
        print("board:")
        for row in self.__board:
            print(" ".join(map(str, row)))
        print("__dict:")
        for entity, coords in self.__dict.items():
            print(f"{entity}({len(coords)}): {coords}")

    def is_solution(self):
        for coords in self.__dict.values(): 
            for i, (row1, col1) in enumerate(coords): 
                dis = self.__bfs(row1, col1, self.__n, self.__board) #least-turn 2D array
                for row2, col2 in coords[i+1:]:
                    if dis[row2][col2] <= 2: 
                        return True
        return False
    
    # Check whether valid path (<=2 turns) exists
    def is_path(self, row1, col1, row2, col2):
        dis = self.__bfs(row1, col1, self.__n, self.__board)
        if dis[row2][col2]<=2:
            return True
        return False
    
    def has_item(self):
        res = False
        for entity_list in self.__dict.values():
            if len(entity_list):
                res = True
        return res

    # Delete pair as specified. Return True if suceed, False if wrong colors
    def del_pair(self, row1, col1, row2, col2): 
        entity1 = self.__board[row1][col1]
        entity2 = self.__board[row2][col2]

        if entity1 != entity2:
            return False
        
        print(f"removing {(row1, col1)} from: {self.__dict[entity1]}")
        print(f"removing {(row2, col2)} from: {self.__dict[entity2]}\n")

        self.__dict[entity1].remove((row1, col1))
        self.__dict[entity2].remove((row2, col2))
        self.__board[row1][col1] = 0
        self.__board[row2][col2] = 0

        self.__update_game()
        return True

    def __bfs(self, row, col, n, g):
        q = deque([[row, col]])
        dis = np.full((n, n), self.__n, dtype=int) #turns
        vis = np.zeros((n, n), dtype=int) #visited
        dis[row][col] = -1
        vis[row][col] = 1

        while len(q):
            ur, uc = q.popleft() #u_row, u_col
            for rr, rc in ROTATION.values():
                for i in range(1, n):
                    vr, vc = ur+i*rr, uc+i*rc
                    if vr<0 or vc<0 or vr>=n or vc>=n or vis[vr][vc]:
                        break
                    dis[vr][vc]=dis[ur][uc]+1
                    vis[vr][vc]=1
                    if g[vr][vc]!=0: 
                        break
                    q.append((vr, vc))
        return dis

        

