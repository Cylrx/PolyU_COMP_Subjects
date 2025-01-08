package utils.validator;

public interface Validator {
    boolean eval(String str);
    String reason();
}

