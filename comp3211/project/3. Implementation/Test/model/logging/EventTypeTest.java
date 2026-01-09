package model.logging;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EventTypeTest {
    @Test
    public void testToString() {
        assertEquals("Captured", EventType.Captured.toString());
        assertEquals("Move", EventType.Move.toString());
        assertEquals("Withdraw", EventType.Withdraw.toString());
    }

    @Test
    public void testAllValues() {
        EventType[] types = EventType.values();
        assertEquals(3, types.length);
        assertTrue(contains(types, EventType.Captured));
        assertTrue(contains(types, EventType.Move));
        assertTrue(contains(types, EventType.Withdraw));
    }

    private boolean contains(EventType[] array, EventType value) {
        for (EventType type : array) {
            if (type == value) {
                return true;
            }
        }
        return false;
    }
}

