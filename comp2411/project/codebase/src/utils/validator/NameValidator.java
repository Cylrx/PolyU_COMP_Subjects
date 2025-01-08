package utils.validator;

/**
 * Validates a name component (either first or last name).
 */
public class NameValidator implements Validator {
    /**
     * The name component is valid only if the following conditions are met:
     * 1. The name component is not empty
     * 2. The name component contains only alphabetic characters
     * 3. The name component is at most 20 characters long
     * 4. The name component does not contain any special characters
     * 5. First character of the name component is capitalized
     * @param str the name component to validate
     * @return true if the name component is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {
        boolean res = !str.isEmpty() && str.length() <= 100;
        res = res && str.matches("^[a-zA-Z]+$");
        res = res && Character.isUpperCase(str.charAt(0));
        return res;
    }

    @Override
    public String reason() {
        return "Name must contain only alphabetic characters, at most 20 characters long, capitalized";
    }
}
