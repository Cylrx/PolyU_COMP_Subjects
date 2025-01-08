package utils.validator;

public class AccountTypeValidator implements Validator {
    /**
     * The account type can only be one of the followings:
     * - Student
     * - Staff
     * - Alumni
     * - Guest
     * @param str the account type to validate
     * @return true if the account type is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {
        return str.equals("Student") || str.equals("Staff") || str.equals("Alumni") || str.equals("Guest");
    }

    @Override
    public String reason() {
        return "Account type must be one of the following: Student, Staff, Alumni";
    }
}
