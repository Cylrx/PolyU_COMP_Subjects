package utils.validator;

public class EmailValidator implements Validator {
    /**
     * The email is valid only if the following conditions are met:
     * 1. The email contains only alphanumeric characters, @, ., +, _, -
     * 2. The email contains exactly one @ character
     * 3. The email contains at least one . character after the @ character
     * @param str the email to validate
     * @return true if the email is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {
        return str.matches("^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$") && str.length() <= 100;
    }

    @Override
    public String reason() {
        return "Email must contain only alphanumeric characters, @, ., +, _, -, with maximum length of 100";
    }
}
