package model;

import model.elements.PieceType;

import java.io.Serializable;

public class Piece implements Serializable {
    private final PieceType type;
    private Square curPosition;
    private boolean status; // 0 for died, 1 for alive
    private final boolean belongs; // 0 for lower player, 1 for upper player

    public Piece(PieceType _type, boolean _belongs) {
        type = _type;
        belongs = _belongs;
        status = true;
    }

    public boolean getStatus() {
        return status;
    }

    public PieceType getType() {
        return type;
    }

    public boolean getBelongs() {
        return belongs;
    }

    public void setPosition(Square position) {
        curPosition = position;
    }

    public void setDie() {
        status = false;
        curPosition = null;
    }

    public void setAlive() {
        status = true;
    }

    public Square getPosition() {
        return curPosition;
    }
}
