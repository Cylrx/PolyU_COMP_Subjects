package controller;

import service.*;
import exceptions.BMSException;
import utils.SecurityUtils;
import view.View;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;

import static utils.InputUtils.*;

public class Attendee implements User {
    private final Menu menu;
    private final char[] passwd;
    private final int ID;

    public Attendee(String email, char[] passwd) throws BMSException {
        this.passwd = passwd;
        try (ResultSet rs = Tables.ATTENDEE.query(new String[]{AttendeeAttr.ATT_ID.getAttrName()}, AttendeeAttr.EMAIL.getAttrName() + " = ?", new String[]{email})) {
            if (rs.next()) ID = rs.getInt(1);
            else throw new BMSException("Attendee not found");
        } catch (SQLException e) {
            throw new BMSException("Unexpected SQL error occurred");
        }
        this.menu = new AttendeeMainMenu(ID);
    }

    public void login() throws BMSException, SQLException {
        Attr ID = AttendeeAttr.ATT_ID;
        Attr PW = AttendeeAttr.PASSWORD;
        if (User.auth(ID, PW, this.ID, passwd, false)) menu.start();
        else View.displayError("Password incorrect");
    }

    @Override
    public void close() {
        Arrays.fill(passwd, '\0');
    }
}

class AttendeeMainMenu implements Menu {
    static Menu menu1, menu2, menu3, menu4;

    AttendeeMainMenu(int ID) {
        menu1 = new AttendeeUpdateInfo(ID);
        menu2 = new AttendeeShowBanquets(ID);
        menu3 = new AttendeeSignUp(ID);
        menu4 = new AttendeeSearch(ID);
    }


    @Override
    public void start() throws SQLException{
        String options = """
                1. Update Personal Info
                2. Show Available Banquets
                3. Register a Banquet
                4. Search Registration
                0. Logout
                """;
        boolean isRunning = true;
        while (isRunning) {
            View.displayOptions("Action", options);
            int op = getDigit("");
            switch (op) {
                case 1 -> menu1.start();
                case 2 -> menu2.start();
                case 3 -> menu3.start();
                case 4 -> menu4.start();
                case 0 -> isRunning = false;
                default -> View.displayBadInput("{1, 2, 3, 0}", op);
            }
        }
        View.displayMessage("Logged out");
    }
}

class AttendeeUpdateInfo implements Menu {
    private final int ID;

    AttendeeUpdateInfo(int ID) {
        this.ID = ID;
    }

    @Override
    public void start() throws SQLException{
        String options = """
                1. Update Email
                2. Update Password
                3. Update First Name
                4. Update Last Name
                5. Update Organization
                6. Update Type
                7. Update Address
                8. Update Phone
                0. Go Back
                """;
        AttendeeAttr[] attrs = AttendeeAttr.values();

        while (true) {
            View.displayOptions("Info to update", options);
            int op = getDigit("Action");
            switch (op) {
                case 0 -> { return; }
                case 1 -> updateId(ID);
                case 2 -> updatePw(ID);
                case 3, 4, 5, 6, 7, 8 -> {
                    String val = attrs[op].inputNewVal();
                    String conditionClause = AttendeeAttr.ATT_ID + " = ?";
                    String[] conditionVals = new String[]{String.valueOf(ID)};
                    attrs[op].updateTo(val, conditionClause, conditionVals);
                }
            }
        }
    }

    private void updateId(int ID) throws SQLException{
        // Assumes CASCADE UPDATE is on
        String newId = AttendeeAttr.EMAIL.inputUniqueVal();
        String conditionClause = AttendeeAttr.ATT_ID.getAttrName() + " = ?";
        String[] conditionVals = new String[]{String.valueOf(ID)};
        AttendeeAttr.EMAIL.updateTo(newId, conditionClause, conditionVals);
    }

    private void updatePw(int ID) throws SQLException {
        char[] newPw = getNewPasswd("New password");
        String conditionClause = AttendeeAttr.ATT_ID.getAttrName() + " = ?";
        String[] conditionVals = new String[]{String.valueOf(ID)};
        String hashPw = SecurityUtils.toHash(newPw);
        AttendeeAttr.PASSWORD.updateTo(hashPw, conditionClause, conditionVals);
    }
}

class AttendeeSignUp implements Menu {
    private final int ID;
    AttendeeSignUp(int ID) {this.ID = ID;}

    @Override
    public void start() throws SQLException{
        // input valid BIN
        String banquetID = BanquetAttr.BIN.inputHasVal();
        try(ResultSet rs = Tables.REGISTRATION.query(new String[]{}, BanquetAttr.BIN.getAttrName() + " = ? AND Att_ID = ?", new String[]{banquetID, Integer.toString(ID)})) {
            if (rs.next()) {
                View.displayError("already registered to this banquet");
                return;
            }
        }

        // check for availability
        String[] columns = new String[]{};
        String[] conditionVals = new String[]{banquetID, "1", "0"};
        String conditionClause = BanquetAttr.BIN + " = ? AND " + BanquetAttr.AVAILABILITY + " = ? AND " + BanquetAttr.QUOTA + " > ?";
        try (ResultSet rs = Tables.BANQUET.query(columns, conditionClause, conditionVals)) {
            if (!rs.next()) {
                View.displayError("banquet not available or insufficient quota");
                return;
            }
        }
        View.displayMessage("Available meals:");
        AdminEditBanquet.showMeals(banquetID);

        String mealID = MealAttr.DISH_NAME.inputHasVal();
        conditionVals = new String[]{banquetID, mealID};
        conditionClause = RegistrationAttr.BIN + " = ? AND " + RegistrationAttr.DISH_NAME + " = ?";
        try (ResultSet rs = Tables.MEAL.query(columns, conditionClause, conditionVals)) {
            if (!rs.next()) {
                View.displayError("Meal not available in this banquet");
                return;
            }
        }
        RegistrationAttr[] otherAttrs = new RegistrationAttr[]{RegistrationAttr.DRINK, RegistrationAttr.SEAT, RegistrationAttr.REMARKS};
        String[] otherVals = Attr.inputNewVals(otherAttrs);
        String[] vals = new String[]{Integer.toString(ID), banquetID, mealID, otherVals[0], otherVals[1], "0", otherVals[2]};

        try {
            Tables.REGISTRATION.insert(vals);
        } catch (BMSException e) {
            // cannot happen, because already checked for not registered BIN
        }

        BanquetAttr.changeQuota(-1, banquetID);
    }
}

class AttendeeShowBanquets implements Menu {
    private final int ID;
    AttendeeShowBanquets(int ID) {this.ID = ID;}
    public void start() throws SQLException {
        String[] header = {"BIN","Name","Date","Time"};
        ArrayList<String[]> rows = new ArrayList<>();
        int colNum = header.length;

        String[] columns = new String[]{};
        String conditionClause = BanquetAttr.AVAILABILITY.getAttrName() + " = ? AND " + BanquetAttr.QUOTA.getAttrName() + "> ? AND NOT EXISTS (SELECT * FROM Registration WHERE Registration.BIN = Banquet.BIN AND Registration.Att_ID = ?)";
        String[] conditionVals = new String[]{"1","0", Integer.toString(ID)};

        try(ResultSet rs = Tables.BANQUET.query(columns, conditionClause, conditionVals)) {
            while (rs.next()) {
                String[] row = new String[colNum];
                for (int i = 0; i < colNum; i++)
                    row[i] = rs.getString(i + 1);
                rows.add(row);
            }
            if (rows.isEmpty())
                View.displayMessage("No banquets available");
            else View.displayTable(header, rows);
        }

    }
}

class AttendeeSearch implements Menu{
    private final int ID;
    AttendeeSearch(int ID) {this.ID = ID;}

    @Override
    public void start() throws SQLException {
        String criteria = getStr("Select criteria to search by (Date or Name)");
        String val1, val2;
        switch (criteria) {
            case "Date" -> {
                val1 = BanquetAttr.DATE.inputNewVal("start date");
                val2 = BanquetAttr.DATE.inputNewVal("end date");
            }
            case "Name" -> {
                val1 = getStr("a part of the banquet name");
                val2 = "";
            }
            default -> {
                View.displayBadInput("Date or Name", criteria);
                return;
            }
        }
        String[] headers = Attr.getDescriptions(RegistrationAttr.values());
        headers = Arrays.copyOfRange(headers, 1, headers.length);

        ArrayList<String[]> rows = new ArrayList<>();
        int colNum = headers.length;
        String[] columns = Attr.getColumns(RegistrationAttr.values());
        columns = Arrays.copyOfRange(columns, 1, columns.length);
        String conditionClause = RegistrationAttr.ATT_ID.getAttrName() + " = ?";
        String[] conditionVals = new String[]{Integer.toString(ID)};
        try (ResultSet rs = Tables.REGISTRATION.query(columns, conditionClause, conditionVals)) {
            while (rs.next()) {
                String[] row = new String[colNum];
                for (int i = 0; i < colNum; i++)
                    row[i] = rs.getString(i + 1);
                switch(criteria){
                    case "Date" -> {
                        try(ResultSet trs = Tables.BANQUET.query(new String[]{BanquetAttr.DATE.getAttrName()}, BanquetAttr.BIN.getAttrName() + " = ?", new String[]{row[0]})) {
                            if (trs.next()) {
                                String date = trs.getString(1);
                                if (date.compareTo(val1) < 0 || date.compareTo(val2) > 0)
                                    continue;
                            }
                        }
                    }
                    case "Name" -> {
                        try(ResultSet trs = Tables.BANQUET.query(new String[]{BanquetAttr.BANQUET_NAME.getAttrName()}, BanquetAttr.BIN.getAttrName() + " = ?", new String[]{row[0]})) {
                            if (trs.next()) {
                                String name = trs.getString(1);
                                if (!name.contains(val1))
                                    continue;
                            }
                        }
                    }
                }
                rows.add(row);
            }
            if (rows.isEmpty())
                View.displayMessage("No registrations found");
            else View.displayTable(headers, rows);
        }
    }
}