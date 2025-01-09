from model import GameStatus
from enum import Enum, auto
import pygame.gfxdraw
import control

SCREEN_SIZE = (1280, 720)
SCREEN_COL = pygame.Color(78, 166, 133)
TILE_COL_LIGHT = pygame.Color(191, 216, 189)
TILE_COL_DARK = pygame.Color(152, 201, 163)
HIGHLIGHT_COL = pygame.Color(245, 245, 224, 255)
HIGHLIGHT_THICKNESS = 3 # 0 = no highlight
IMG_SHRINK = 1
BUTTON_SCALE = 0.25
BUTTON_SEPARATION = 75
MARGIN = 10

class SELECTION(Enum):
    DUPLICATE = auto()
    NOT_ENOUGH = auto()
    OUT_OF_BOUND = auto()
    LIST_FULL = auto()
    WRONG_PAIR = auto()
    TOO_MUCH = auto()
    NO_PATH = auto()
    EMPTY = auto()
    VALID = auto()
    QUIT = auto()
    

class GameWindow:
    hard_surface = pygame.image.load("./diff/hard.png")
    middle_surface = pygame.image.load("./diff/middle.png")
    easy_surface = pygame.image.load("./diff/easy.png")

    diff_surface = [easy_surface, middle_surface, hard_surface]
    diff_num = ["EASY", "MIDDLE", "HARD"]

    def __init__(self, logic):
        pygame.init()
        self.__log("Pygame Initialized")
        self.logic = logic
        self.logic.subscribe(self)
        self.__log("Subscribed to GameLogic")
        self.__screen = pygame.display.set_mode(SCREEN_SIZE)
        self.__clock = pygame.time.Clock()
        self.__n = 0
        self.__m = 0
        self.__board = None
        self.__dict = dict()
        self.__status = tuple()
        self.__selected = []
        for i in range(3):
            self.diff_surface[i] = pygame.transform.scale_by(self.diff_surface[i], BUTTON_SCALE)

    def __log(self, message: str):
        print(f"[View]: {message}")

    def update(self, info):
       self.__n, self.__m, self.__board, self.__dict, self.__status = info 
       print("[View]: Game Upate Recieved")

    def __render_game(self):
        # Render Board
        self.__screen.fill(SCREEN_COL)
        mouse_col, mouse_row = control.get_mouse_pos(self.__n)
        for row in range(self.__n):
            for col in range(self.__n):
                color = TILE_COL_LIGHT if (row+col)%2 else TILE_COL_DARK
                pygame.gfxdraw.box(self.__screen, calc_pos(self.__n, row, col, 0), color)
                if mouse_row == row and mouse_col == col:
                    self.__draw_highlight(HIGHLIGHT_THICKNESS, HIGHLIGHT_COL, row, col, False)
        
        # Render Entities
        for entity, entity_list in self.__dict.items():
            for row, col in entity_list:
                x, y, square_width, _ = calc_pos(self.__n, row, col, IMG_SHRINK)
                img_path = "./img/character_" + str(entity) + ".png"
                img_surface = pygame.image.load(img_path)
                img_width, _ = img_surface.get_size()
                img_surface = pygame.transform.scale_by(img_surface, (square_width-IMG_SHRINK*3)/img_width)
                self.__screen.blit(img_surface, (x+IMG_SHRINK, y+IMG_SHRINK))
    
        # Render Selected Tiles
        for tile in self.__selected:
            row, col = tile
            self.__draw_highlight(HIGHLIGHT_THICKNESS, HIGHLIGHT_COL, row, col, True)
    
    def __draw_highlight(self, thickness, color, row, col, is_gradient):
        for i in range(thickness):
            if is_gradient:
                color = pygame.Color(color.r, color.g, color.b, color.a-i*10)
            pygame.gfxdraw.rectangle(self.__screen, calc_pos(self.__n, row, col, i), color)
    
    def check_pair(self):
        s1, s2 = self.__selected
        print(f"checking pair: {s1}, {s2}")
        if not self.logic.is_path(*s1, *s2):
            return SELECTION.NO_PATH
        if not self.logic.del_pair(*s1, *s2):
            return SELECTION.WRONG_PAIR
        return SELECTION.VALID

    def __render_menu(self):
        for i, surface in enumerate(self.diff_surface):
            self.__screen.blit(surface, (MARGIN, MARGIN+i*BUTTON_SEPARATION))
    
    def diff_clicked(self):
        x, y= pygame.mouse.get_pos()
        for i, surface in enumerate(self.diff_surface):
            if x >= MARGIN and x < MARGIN+surface.get_width():
                if y>=MARGIN+i*BUTTON_SEPARATION and y < MARGIN+i*BUTTON_SEPARATION+surface.get_height():
                    return i
        return -1

    def __display_win(self):
        screen_width, screen_height = SCREEN_SIZE
        win_surface = pygame.image.load("./diff/win.png")
        center_x = (screen_width - win_surface.get_rect().width) // 2
        center_y = (screen_height - win_surface.get_rect().height) // 2
        self.__screen.blit(win_surface, (center_x, center_y))

    def game_loop(self):
        running = True
        while running: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            status=self.__status
            if status == GameStatus.RUNNING:
                if pygame.mouse.get_pressed()[0]:
                    diff = self.diff_clicked()
                    if diff>=0:
                        self.__status = GameStatus.IDLE
                        self.__log(f"Restarted with {diff}")
                        return diff

                    control.handle_click(self, self.__n, self.__selected, self.__board)
                if pygame.key.get_pressed()[pygame.K_RETURN]:
                    control.handle_enter(self, self.__selected)
                    self.__selected.clear()
                if not self.logic.is_solution() or not self.logic.has_item():
                    self.__status = GameStatus.END
                self.__render_game()
                self.__render_menu()
            elif status == GameStatus.END:
                self.__display_win()

            pygame.display.flip()
            self.__clock.tick(60)
        pygame.quit()



def calc_pos(n, row, col, offset):
    w, h = SCREEN_SIZE
    square_width = (h-2*MARGIN)//n
    x = w/2 - square_width*(n/2-col)
    y = MARGIN + square_width*row
    return x+offset, y+offset, square_width-offset*2, square_width-offset*2
