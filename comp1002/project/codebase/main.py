from model import GameLogic
from view import GameWindow

DIFFICULTY = [[10, 5], [10, 10], [20, 50]]

def main():
    diff = 0
    logic = GameLogic()
    window = GameWindow(logic)
    while True:
        logic.new_game(*DIFFICULTY[diff])
        logic.print_all()
        diff = window.game_loop()
        print("restarted")

if __name__ == "__main__":
    main()