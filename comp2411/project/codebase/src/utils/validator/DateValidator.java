package utils.validator;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

public class DateValidator implements Validator {
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE;
    private static final LocalDate LOWER_BOUND = LocalDate.of(2024, 1, 1);
    private static final LocalDate UPPER_BOUND = LocalDate.of(9999, 12, 31);

    /**
     * The date is valid only if the following conditions are met:
     * 1. The date is in the format of yyyy-MM-dd
     * 2. The date is on or after 2024-01-01
     * 3. The date is before 2027-01-01
     * @param str the date string to validate
     * @return true if the date is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {
        if (str == null) return false;
        try {
            LocalDate date = LocalDate.parse(str, DATE_FORMATTER);
            return date.isEqual(LOWER_BOUND) || date.isEqual(UPPER_BOUND) || date.isAfter(LOWER_BOUND) && date.isBefore(UPPER_BOUND);
        } catch (DateTimeParseException e) {
            return false;
        }
    }

    @Override
    public String reason() {
        return "Date must be in the format of yyyy-MM-dd, between 2024-01-01 and before 9999-12-31";
    }
}
