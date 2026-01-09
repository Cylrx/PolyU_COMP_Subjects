package model.elements;

import java.io.Serializable;

public class Coordinate implements Serializable {
    private int x;
    private int y;

    public Coordinate(int _x, int _y) {
        x = _x;
        y = _y;
    }

    public Coordinate(Coordinate c) {
        x = c.x;
        y = c.y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public void changeX(int delta) {
        x += delta;
    }

    public void changeY(int delta) {
        y += delta;
    }

    public boolean checkBound(int boundX, int boundY) {
        return x > 0 && x <= boundX && y > 0 && y <= boundY;
    }

    public boolean equals(Object obj) {
        if(obj instanceof Coordinate target) {
            return x == target.getX() && y == target.getY();
        } else {
            return false;
        }
    }

    public void moveByChar(char direction) {
        switch (direction) {
            case 'L':
                changeX(-1);
                break;
            case 'R':
                changeX(1);
                break;
            case 'U':
                changeY(1);
                break;
            case 'D':
                changeY(-1);
                break;
        }
    }

    public String toString() {
        return "(" + ((char)(x - 1 + 'A')) + ", " + y + ")";
    }
}
