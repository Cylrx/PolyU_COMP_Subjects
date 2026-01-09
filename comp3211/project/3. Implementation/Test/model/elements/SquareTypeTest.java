package model.elements;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SquareTypeTest {
    @Test
    public void testToString() {
        assertEquals("normal", SquareType.Normal.toString());
        assertEquals("river", SquareType.River.toString());
        assertEquals("trap", SquareType.Trap.toString());
        assertEquals("den", SquareType.Den.toString());
    }

    @Test
    public void testAllValues() {
        SquareType[] types = SquareType.values();
        assertEquals(4, types.length);
        assertTrue(contains(types, SquareType.Normal));
        assertTrue(contains(types, SquareType.River));
        assertTrue(contains(types, SquareType.Trap));
        assertTrue(contains(types, SquareType.Den));
    }

    private boolean contains(SquareType[] array, SquareType value) {
        for (SquareType type : array) {
            if (type == value) {
                return true;
            }
        }
        return false;
    }
}

