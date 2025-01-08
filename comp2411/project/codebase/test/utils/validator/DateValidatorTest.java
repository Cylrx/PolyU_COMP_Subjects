package utils.validator;

import static org.junit.Assert.*;
import org.junit.Test;

public class DateValidatorTest {
    private static final DateValidator dv = new DateValidator();

    @Test
    public void testEval() {
        // Valid dates
        assertTrue(dv.eval("2024-10-05"));
        assertTrue(dv.eval("2026-12-31"));
        assertTrue(dv.eval("2024-02-29"));
        assertTrue(dv.eval("2100-12-31"));
        assertTrue(dv.eval("2027-01-01"));
        assertTrue(dv.eval("2101-01-01"));

        // Invalid leap year dates
        assertFalse(dv.eval("2023-02-29"));
        assertFalse(dv.eval("2021-02-29"));

        // Invalid month
        assertFalse(dv.eval("2023-13-01"));
        assertFalse(dv.eval("2023-00-10"));

        // Invalid day
        assertFalse(dv.eval("2023-04-31"));
        assertFalse(dv.eval("2023-06-31"));
        assertFalse(dv.eval("2023-12-32"));

        // Invalid format
        assertFalse(dv.eval("2023-6-15"));
        assertFalse(dv.eval("15-06-2023"));
        assertFalse(dv.eval("2023/06/15"));
        assertFalse(dv.eval("2023.06.15"));
        assertFalse(dv.eval("06-15-2023"));
        assertFalse(dv.eval("2023-001-01"));
        assertFalse(dv.eval("2023-01-1"));
        assertFalse(dv.eval("202-01-01"));
        assertFalse(dv.eval("023-01-01"));
        assertFalse(dv.eval("2023-1-1"));
        assertFalse(dv.eval("20230101"));
        assertFalse(dv.eval("2023-01"));
        assertFalse(dv.eval("2023"));
        assertFalse(dv.eval(""));

        // Non-numeric characters
        assertFalse(dv.eval("abcd-ef-gh"));
        assertFalse(dv.eval("2023-aa-01"));
        assertFalse(dv.eval("20b3-01-01"));
        assertFalse(dv.eval("2023-0a-01"));
        assertFalse(dv.eval("2023-01-a1"));

        // Out of bounds

        assertFalse(dv.eval("-2023-01-01"));
        assertFalse(dv.eval("2023-12-31"));
        assertFalse(dv.eval("99999-12-31"));

        // Null and whitespace
        assertFalse(dv.eval(null));
        assertFalse(dv.eval(" "));
        assertFalse(dv.eval("    "));
        assertFalse(dv.eval("2023-01-01 "));
        assertFalse(dv.eval(" 2023-01-01"));
        assertFalse(dv.eval("2023 -01-01"));
        assertFalse(dv.eval("2023- 01-01"));
        assertFalse(dv.eval("2023-01 -01"));
        assertFalse(dv.eval("2023-01- 01"));
    }
}