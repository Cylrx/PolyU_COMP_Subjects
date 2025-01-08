package utils.validator;

import org.junit.Test;

import static org.junit.Assert.*;

public class TimeValidatorTest {
    private static final TimeValidator tv = new TimeValidator();

    @Test
    public void testEval() {
        // Valid times
        assertTrue(tv.eval("00:00:00"));
        assertTrue(tv.eval("12:30:45"));
        assertTrue(tv.eval("23:59:59"));
        assertTrue(tv.eval("01:01:01"));
        assertTrue(tv.eval("15:20:35"));
        assertTrue(tv.eval("07:08:09"));

        // Invalid hour
        assertFalse(tv.eval("24:00:00"));
        assertFalse(tv.eval("-01:00:00"));
        assertFalse(tv.eval("25:00:00"));
        assertFalse(tv.eval("99:99:99"));

        // Invalid minute
        assertFalse(tv.eval("12:60:00"));
        assertFalse(tv.eval("12:-1:00"));
        assertFalse(tv.eval("12:99:00"));

        // Invalid second
        assertFalse(tv.eval("12:30:60"));
        assertFalse(tv.eval("12:30:-1"));
        assertFalse(tv.eval("12:30:99"));

        // Invalid format
        assertFalse(tv.eval("12:30"));
        assertFalse(tv.eval("123456"));
        assertFalse(tv.eval("12:30:45.678"));
        assertFalse(tv.eval("ab:cd:ef"));
        assertFalse(tv.eval(null));
        assertFalse(tv.eval(""));
        assertFalse(tv.eval(" 12:30:45"));
        assertFalse(tv.eval("12:30:45 "));
        assertFalse(tv.eval("12 :30:45"));
        assertFalse(tv.eval("12: 30:45"));
        assertFalse(tv.eval("12:30 :45"));
        assertFalse(tv.eval("12:30: 45"));
        assertFalse(tv.eval("1:2:3"));
        assertFalse(tv.eval("01:2:03"));
        assertFalse(tv.eval("1:02:03"));
        assertFalse(tv.eval("01:02:3"));
        assertFalse(tv.eval("01-02-03"));
        assertFalse(tv.eval("012345"));
        assertFalse(tv.eval("::"));
        assertFalse(tv.eval("..:..:.."));
    }
}