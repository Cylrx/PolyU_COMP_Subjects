package service;

import utils.validator.*;

import java.sql.ResultSet;
import java.sql.SQLException;

import static controller.Controller.db;

public enum AttendeeAttr implements Attr{
    ATT_ID("Att_ID", "Attendee ID", null),
    EMAIL("Email", "Email", new EmailValidator()),
    PASSWORD("Password", "Password", null),
    FIRST_NAME("First_name", "First name", new NameValidator()),
    LAST_NAME("Last_name", "Last name", new NameValidator()),
    ORGANIZATION("Organization", "Organization", new OrganizationValidator()),
    ATTENDEE_TYPE("Attendee_type", "Account type", new AccountTypeValidator()),
    ADDRESS("Address", "Address", new AddressValidator()),
    PHONE("Mobile_num", "Phone number", new PhoneValidator());

    private final String attrName;
    private final String description; // user friendly description
    private final Validator vd;

    private static final Tables table = Tables.ATTENDEE;

    AttendeeAttr(String attrName, String description, Validator vd) {
        this.attrName = attrName;
        this.description = description;
        this.vd = vd;
    }

    @Override public Tables getTable() { return table; }
    @Override public String getAttrName() { return attrName; }
    @Override public String getDescription() { return description; }
    @Override public Validator getValidator() { return vd;}
    @Override public boolean isUpdatable() { return this != ATT_ID; }
    public static int getID(String email) throws SQLException {
        int ret = 0;
        try (ResultSet rs = db.executeQuery("SELECT Att_ID FROM Attendee WHERE Email = ?", email)) {
            if (rs.next()) ret =  rs.getInt(1);
        }
        return ret;
    }
}
