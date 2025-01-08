package utils.validator;

public class IntValidator implements Validator {
    @Override
    public boolean eval(String str) {
        try {
            Integer.parseInt(str);
        } catch (NumberFormatException e) {
            return false;
        }
        return true;
    }

    @Override
    public String reason() {
        return "Must be an integer";
    }
}
