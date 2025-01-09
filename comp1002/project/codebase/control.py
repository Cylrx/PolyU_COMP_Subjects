import view
import pygame

def get_mouse_pos(n):
    x, y = pygame.mouse.get_pos()
    dx, dy, square_width, _ = view.calc_pos(n, 0, 0, 0)
    return int((x-dx)//square_width), int((y-dy)//square_width)

def handle_click(window, n, selected, board):
    col, row = get_mouse_pos(n)
    if len(selected)>=2:
        return view.SELECTION.LIST_FULL
    if row < 0 or col < 0 or row >= n or col >= n:
        return view.SELECTION.OUT_OF_BOUND
    if (row, col) in selected:
        return view.SELECTION.DUPLICATE
    if board[row][col] == 0:
        return view.SELECTION.EMPTY
   
    selected.append((row, col))
    return view.SELECTION.VALID

def handle_enter(window, selected):
    print("handling enter")
    selected = list(set(selected))
    if len(selected)<2:
        return view.SELECTION.NOT_ENOUGH
    if len(selected)>2:
        return view.SELECTION.TOO_MUCH
    return window.check_pair()


    

