package utils.validator;

public class PhoneValidator implements Validator {
    /**
     * The phone number is valid only if the following conditions are met:
     * 1. The phone number is exactly 8 digits long
     * 2. The phone number contains only digits
     * @param str the phone number to validate
     * @return true if the phone number is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {
        return str.matches("^[0-9]{8}$");
    }

    @Override
    public String reason() {
        return "Phone number must contain exactly 8 digits";
    }
}
