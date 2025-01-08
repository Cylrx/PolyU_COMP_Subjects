package utils.validator;

public class SeatValidator implements Validator{
    @Override
    public boolean eval(String str) {
        return str.matches("[A-Z][0-9]{1,2}");
    }

    @Override
    public String reason() {
        return "Seat must be an uppercase letter followed by 1 or 2 digits";
    }
}
