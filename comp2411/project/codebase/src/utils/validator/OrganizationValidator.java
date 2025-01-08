package utils.validator;

public class OrganizationValidator implements Validator {
    /**
     * The organization can only be one of the followings:
     * - PolyU
     * - SPEED
     * - HKCC
     * @param str the organization to validate
     * @return true if the organization is valid, false otherwise
     */
    @Override
    public boolean eval(String str) {return str.matches("[A-Za-z0-9 ]{1,100}");}

    @Override
    public String reason() {
        return "Organization must be alphanumeric and less than 100 characters";
    }
}
