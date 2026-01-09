package model.elements;

public enum PieceType {
    Rat("rat"),
    Cat("cat"),
    Dog("dog"),
    Wolf("wolf"),
    Leopard("leopard"),
    Tiger("tiger"),
    Lion("lion"),
    Elephant("elephant");

    private final String name;

    private PieceType(String _name) {
        name = _name;
    }

    public boolean canCapture(PieceType target) {
        if (this == PieceType.Rat) {
            return target == PieceType.Rat || target == PieceType.Elephant;
        } else if(this == PieceType.Elephant) {
            return target != PieceType.Rat;
        } else {
            return this.ordinal() >= target.ordinal();
        }
    }

    public boolean canLeap() {
        return this == PieceType.Lion || this == PieceType.Tiger;
    }

    public String toString() {
        return name;
    }
}
