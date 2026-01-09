package model;

import model.elements.Coordinate;
import model.elements.SquareType;

public class SpecialSquare extends Square {
    private boolean belongs; // 0 for lower player, 1 for upper player;

    public SpecialSquare(Coordinate coordinate, SquareType type, boolean _belongs) {
        super(coordinate, type);
        belongs = _belongs;
    }

    public boolean getSide() {
        return belongs;
    }
}
