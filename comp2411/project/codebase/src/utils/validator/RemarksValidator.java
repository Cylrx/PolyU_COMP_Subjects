package utils.validator;

public class RemarksValidator implements Validator{
    @Override
    public boolean eval(String str) {
        return str.matches("[A-Za-z0-9 ]{1,100}");
    }

    @Override
    public String reason() {
        return "Remarks must be alphanumeric and less than 100 characters";
    }
}
