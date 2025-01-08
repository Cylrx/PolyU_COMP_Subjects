package utils.validator;

public class BoolValidator implements Validator {
    @Override
    public boolean eval(String str) {
        return str.equals("1") || str.equals("0");
    }

    @Override
    public String reason() {
        return "Availability must be either 1 or 0";
    }
}
