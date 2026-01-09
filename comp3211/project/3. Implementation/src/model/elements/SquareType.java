package model.elements;

public enum SquareType {
    Normal("normal"),
    River("river"),
    Trap("trap"),
    Den("den");

    private final String type;

    private SquareType(String _type) {
        type = _type;
    }

    public String toString() {
        return type;
    }
}
