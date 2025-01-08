package service;

import utils.validator.EmailValidator;
import utils.validator.Validator;

public enum AdminAttr implements Attr{
    ADMIN_ID("Admin_ID", "Admin ID", null),
    EMAIL("Email", "Email", new EmailValidator()),
    PASSWORD("Password", "Password", null);

    private final String attrName;
    public final String description; // user friendly description
    private final Validator vd;

    private static final Tables table = Tables.ADMIN;

    AdminAttr(String attrName, String description, Validator vd) {
        this.attrName = attrName;
        this.description = description;
        this.vd = vd;
    }

    @Override public Tables getTable() { return table; }
    @Override public String getAttrName() { return attrName; }
    @Override public String getDescription() { return description; }
    @Override public Validator getValidator() { return vd; }
    @Override public boolean isUpdatable() { return this != ADMIN_ID; }
}
