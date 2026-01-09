package model;

import model.elements.Coordinate;
import model.elements.PieceType;
import model.elements.SquareType;

import java.io.Serializable;
import java.util.HashMap;

public class MovingValidator implements Serializable {
    private final HashMap<String, Piece> upperPlayerPieces;
    private final HashMap<String, Piece> lowerPlayerPieces;
    private final Board board;
    private final char[] directions = {'U', 'D', 'L', 'R'};

    public MovingValidator(Board _board) {
        board = _board;
        upperPlayerPieces = new HashMap<String, Piece>();
        lowerPlayerPieces = new HashMap<String, Piece>();

        // Scan the whole board once at construction time and
        // build two dictionaries:
        //  - lowerPlayerPieces: all pieces that belong to the lower player
        //  - upperPlayerPieces: all pieces that belong to the upper player
        // The key is the piece name string (e.g. "rat", "tiger"), taken from PieceType.toString().
        // Later, when we get a command like "move rat U", we will look up the actual Piece
        // object from these dictionaries.
        int n = board.getLength(), m = board.getWidth();
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++) {
                Piece curPiece = board.getPiece(i, j);
                if(curPiece == null) {
                    continue;
                }
                if(!curPiece.getBelongs()) {
                    lowerPlayerPieces.put(curPiece.getType().toString(), curPiece);
                } else {
                    upperPlayerPieces.put(curPiece.getType().toString(), curPiece);
                }
            }
        }
    }

    public boolean attemptCapture(Piece attacker, Piece defender) {
        // 1. You can never capture your own piece.
        if(attacker.getBelongs() == defender.getBelongs()) {
            return false;
        }

        // 2. River / land constraint:
        //    - Must be both on land or both on river (True/True, False/False)  
        //    - Otherwise capturing is not allowed.
        if(((attacker.getPosition().getType() == SquareType.River) == (defender.getPosition().getType() == SquareType.River))) {
            // 3. Trap rule:
            //    If the defender is standing in a trap of the attacker,
            //    then the attacker can always capture it regardless of rank.
            if(defender.getPosition().getType() == SquareType.Trap && ((SpecialSquare)defender.getPosition()).getSide() != defender.getBelongs()) {
                return true;
            } else {
                // 4. Normal capture rule:
                //    Delegate to PieceType.canCapture, which encodes:
                //      - Rat can capture Rat and Elephant.
                //      - Elephant cannot capture Rat.
                //      - Other animals follow rank order.
                return attacker.getType().canCapture(defender.getType());
            }
        }

        // One piece on river and the other on land ⇒ cannot capture.
        return false;
    }

    /*
        turn should be 0 if this is the lower player's turn
     */
    public Piece getTargetPiece(String name, boolean side) {
        // side == false  → lower player
        // side == true   → upper player
        // The name must match PieceType.toString(), for example "rat", "tiger", etc.
        return (Piece) (side ? upperPlayerPieces.get(name) : lowerPlayerPieces.get(name));
    }

    /*
        turn should be 0 if this is the lower player's turn
     */
    public Coordinate attemptMove(String name, boolean turn, char direction) {
        // Determine which player's piece we are moving based on `turn`.
        // turn == false → lower player's turn
        // turn == true  → upper player's turn
        Piece curPiece = (Piece) (turn ? upperPlayerPieces.get(name) : lowerPlayerPieces.get(name));

        // A dead piece (that has been captured) cannot be moved.
        if(!curPiece.getStatus()) {
            throw new IllegalArgumentException("You can not a piece that has been captured");
        }

        // Start from the current position of the piece and simulate
        // a one‑step move in the given direction (L/R/U/D).
        Coordinate newPosition = new Coordinate(curPiece.getPosition().getCoordinate());
        newPosition.moveByChar(direction);

        // Check board boundary.
        if(!newPosition.checkBound(board.getLength(), board.getWidth())) {
            throw new IllegalArgumentException("Your move out of the board!");
        }

        Square curSquare = board.getSquare(newPosition);

        // Rule: a piece cannot move into its own den.
        if(curSquare.getType() == SquareType.Den && ((SpecialSquare) curSquare).getSide() == curPiece.getBelongs()) {
            throw new IllegalArgumentException("You can not move in your den");
        }

        // If the first target square is a river square, we have special handling
        // for Rat / Lion / Tiger, and forbid other pieces from entering river.
        if(curSquare.getType() == SquareType.River) {
            // Case 1: Rat is the only animal allowed to stay on river squares.
            if(curPiece.getType() == PieceType.Rat) {
                // If the river square is empty, Rat can simply move onto it.
                if(curSquare.getPiece() == null) {
                    return newPosition;
                }
                // If there is a piece on the river square, try to capture it.
                // Note that temptCapture already enforces the river/land and trap rules.
                if(attemptCapture(curPiece, curSquare.getPiece())) {
                    return newPosition;
                } else {
                    // Opponent piece there but not legally capturable.
                    throw new IllegalArgumentException("You can not capture target piece!");
                }

            // Case 2: Lion / Tiger may "jump" over an entire river.
            } else if(curPiece.getType().canLeap()) {
                // We keep moving in the same direction as long as we are within bounds.
                // The idea is:
                //   - All intermediate river squares must be empty (no Rats blocking).
                //   - When we reach the first non‑river square, we may land there
                //     or capture an enemy piece on that square.
                //
                // NOTE: current implementation only returns when the landing square
                //       contains a piece (and capture is allowed). If the first non‑river
                //       square is empty, this loop will keep moving forward until
                //       it leaves the board and finally returns null.
                //       This behavior does NOT match the official Jungle rule and
                //       should be fixed in a future refactoring.
                while(newPosition.checkBound(board.getLength(), board.getWidth())) {
                    curSquare = board.getSquare(newPosition);
                    if(curSquare.getType() == SquareType.River) {
                        // Still in river: check that there is no blocking piece (e.g. Rat).
                        if(curSquare.getPiece() != null) {
                            throw new IllegalArgumentException("You can not cross a river with another piece in the middle");
                        }
                    } else {
                        // We have reached the first non‑river square on the other side.
                        // If there is a piece on this square, try to capture it.
                        if(curSquare.getPiece() != null) {
                            if(attemptCapture(curPiece, curSquare.getPiece())) {
                                return newPosition;
                            } else {
                                // Same‑side piece on landing square.
                                if(curPiece.getBelongs() == curSquare.getPiece().getBelongs()) {
                                    throw new IllegalArgumentException("The target square have another your piece");
                                }
                                // Opponent piece but not legally capturable.
                                throw new IllegalArgumentException("You can not capture target piece!");
                            }
                        } else {
                            return curSquare.getCoordinate();
                        }
                    }
                    newPosition.moveByChar(direction);
                }
            // Case 3: any other animal type is not allowed to enter a river square.
            } else {
                throw new IllegalArgumentException("You can not place move this piece to a river matrix!");
            }
        } else {
            // Non‑river target:
            //   - If empty, the move is always allowed.
            //   - If occupied, we must check whether capture is legal.
            if(curSquare.getPiece() == null) {
                return newPosition;
            }
            if(attemptCapture(curPiece, curSquare.getPiece())) {
                return newPosition;
            } else {
                // Target square has a same‑side piece.
                if(curPiece.getBelongs() == curSquare.getPiece().getBelongs()) {
                    throw new IllegalArgumentException("The target square have another your piece");
                }
                // Opponent piece there but cannot be captured according to rules.
                throw new IllegalArgumentException("You can not capture target piece!");
            }
        }
        return null;
    }

    public boolean haveValidMove(boolean turn) {
        for(PieceType x : PieceType.values()) {
            for(char c : directions) {
                try {
                    attemptMove(x.toString(), turn, c);
                    return true;
                } catch(Exception ignored) {}
            }
        }
        return false;
    }

    // For testing only
    public HashMap<String, Piece> getUpperDictionary() {
        return upperPlayerPieces;
    }

    public HashMap<String, Piece> getLowerDictionary() {
        return lowerPlayerPieces;
    }

}
