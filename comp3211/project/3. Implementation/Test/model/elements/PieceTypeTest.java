package model.elements;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PieceTypeTest {
    private PieceType TypeA, TypeB;

    @Test
    public void testRatCapture() {
        TypeA = PieceType.Rat;
        TypeB = PieceType.Rat;
        assertTrue(TypeA.canCapture(TypeB));

        TypeB = PieceType.Elephant;
        assertTrue(TypeA.canCapture(TypeB));

        TypeB = PieceType.Lion;
        assertFalse(TypeA.canCapture(TypeB));
    }

    @Test
    public void testElephantCapture() {
        TypeA = PieceType.Elephant;
        TypeB = PieceType.Elephant;
        assertTrue(TypeA.canCapture(TypeB));

        TypeB = PieceType.Rat;
        assertFalse(TypeA.canCapture(TypeB));

        TypeB = PieceType.Lion;
        assertTrue(TypeA.canCapture(TypeB));
    }

    @Test
    public void testOtherCapture() {
        TypeA = PieceType.Lion;
        TypeB = PieceType.Dog;
        assertTrue(TypeA.canCapture(TypeB));

        TypeA = PieceType.Cat;
        TypeB = PieceType.Leopard;
        assertFalse(TypeA.canCapture(TypeB));

        TypeA = PieceType.Tiger;
        TypeB = PieceType.Tiger;
        assertTrue(TypeA.canCapture(TypeB));
    }

}