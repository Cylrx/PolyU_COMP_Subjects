package service;

import utils.validator.*;

public enum RegistrationAttr implements Attr {
    ATT_ID(
            AttendeeAttr.ATT_ID.getAttrName(),
            AttendeeAttr.ATT_ID.getDescription(),
            AttendeeAttr.ATT_ID.getValidator()
    ),
    BIN(
            BanquetAttr.BIN.getAttrName(),
            BanquetAttr.BIN.getDescription(),
            BanquetAttr.BIN.getValidator()
    ),
    DISH_NAME(
            MealAttr.DISH_NAME.getAttrName(),
            MealAttr.DISH_NAME.getDescription(),
            MealAttr.DISH_NAME.getValidator()
    ),
    DRINK("Drink", "Drink", new DrinkValidator()),
    SEAT("Seat", "Seat number", new SeatValidator()),
    ATTENDANCE("Attendance", "Attendance", new BoolValidator()), // NUMBER(1)
    REMARKS("Remarks", "Remarks", new RemarksValidator());

    private final String attrName;
    private final String description; // user friendly description
    private final Validator vd;
    private final static Tables table = Tables.REGISTRATION;

    RegistrationAttr(String attrName, String description, Validator vd) {
        this.attrName = attrName;
        this.description = description;
        this.vd = vd;
    }

    @Override public Tables getTable() { return table; }
    @Override public String getAttrName() { return attrName; }
    @Override public String getDescription() { return description; }
    @Override public Validator getValidator() { return vd; }
    @Override public boolean isUpdatable() { return this != ATT_ID && this != BIN && this != DISH_NAME;}
}
