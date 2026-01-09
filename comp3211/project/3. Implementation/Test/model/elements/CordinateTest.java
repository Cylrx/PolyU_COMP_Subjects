package model.elements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CoordinateTest {
    private Coordinate origin;
    private Coordinate destination;

    @BeforeEach
    public void setUp() {
        origin = new Coordinate(0,0);
    }

    @Test
    public void testMoveX() {
        origin.changeX(-5);
        destination = new Coordinate(-5,0);
        assertEquals(origin, destination);
    }

    @Test
    public void testMoveY() {
        origin.changeY(5);
        destination = new Coordinate(0,5);
        assertEquals(origin, destination);
    }

    @Test
    public void testCheckBound() {
        origin = new Coordinate(10,5);
        assertTrue(origin.checkBound(10, 10));

        origin = new Coordinate(10,5);
        assertFalse(origin.checkBound(5, 5));

        origin = new Coordinate(5,5);
        assertTrue(origin.checkBound(5, 5));

        origin = new Coordinate(0,5);
        assertFalse(origin.checkBound(10, 10));
    }

    @Test
    public void testConstructor() {
        Coordinate c1 = new Coordinate(3, 4);
        assertEquals(3, c1.getX());
        assertEquals(4, c1.getY());
    }

    @Test
    public void testCopyConstructor() {
        Coordinate original = new Coordinate(5, 6);
        Coordinate copy = new Coordinate(original);
        assertEquals(original.getX(), copy.getX());
        assertEquals(original.getY(), copy.getY());
        assertTrue(original.equals(copy));
    }

    @Test
    public void testMoveByChar() {
        Coordinate coord = new Coordinate(4, 4);
        
        coord.moveByChar('L');
        assertEquals(3, coord.getX());
        assertEquals(4, coord.getY());
        
        coord.moveByChar('R');
        assertEquals(4, coord.getX());
        assertEquals(4, coord.getY());
        
        coord.moveByChar('U');
        assertEquals(4, coord.getX());
        assertEquals(5, coord.getY());
        
        coord.moveByChar('D');
        assertEquals(4, coord.getX());
        assertEquals(4, coord.getY());
    }

    @Test
    public void testEquals() {
        Coordinate c1 = new Coordinate(3, 4);
        Coordinate c2 = new Coordinate(3, 4);
        Coordinate c3 = new Coordinate(3, 5);
        Coordinate c4 = new Coordinate(4, 4);
        
        assertTrue(c1.equals(c2));
        assertFalse(c1.equals(c3));
        assertFalse(c1.equals(c4));
        assertFalse(c1.equals(null));
        assertFalse(c1.equals("not a coordinate"));
    }

    @Test
    public void testToString() {
        Coordinate c1 = new Coordinate(1, 1);
        String str = c1.toString();
        assertNotNull(str);
        assertTrue(str.contains("("));
        assertTrue(str.contains(")"));
    }

    @Test
    public void testGetXAndGetY() {
        Coordinate coord = new Coordinate(7, 9);
        assertEquals(7, coord.getX());
        assertEquals(9, coord.getY());
    }
}