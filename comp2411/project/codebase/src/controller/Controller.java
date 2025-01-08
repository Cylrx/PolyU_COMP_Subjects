package controller;

import service.DBConnect;
import service.AttendeeAttr;
import service.Attr;
import service.Tables;
import exceptions.BMSException;
import utils.SecurityUtils;
import view.View;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Scanner;

import static utils.InputUtils.*;


public class Controller {
    public static final Scanner sc = new Scanner(System.in);
    public static DBConnect db;
    public static int banquetNum, adminNum, attendeeNum;
    public void init(String url) throws SQLException {
        db = new DBConnect(url);
        String options = """
                1. Attendee
                2. Administrator
                3. New User Registration
                4. Exit
                """;
        boolean isRunning = true;
        ResultSet N = db.executeQuery("SELECT COUNT(*) FROM Banquet");
        N.next();
        banquetNum = N.getInt(1);
        N = db.executeQuery("SELECT COUNT(*) FROM Administrator");
        N.next();
        adminNum = N.getInt(1);
        N = db.executeQuery("SELECT COUNT(*) FROM Attendee");
        N.next();
        attendeeNum = N.getInt(1);
        N.close();
        while (isRunning) {
            View.displayOptions("Account Type", options);
            int op = getDigit("");
            switch (op) {
                case 1 -> loginAttendee();
                case 2 -> loginAdmin();
                case 3 -> registerAttendee();
                case 4 -> {
                    db.close();
                    View.displayExit();
                    isRunning = false;
                }
                default -> View.displayBadInput("{1, 2, 3, 4}", op);
            }
        }
    }

    private void loginAttendee() throws SQLException{
        while (true) {
            String email = getStr("[Attendee] Email");
            char[] password = getPasswd("[Attendee] Password");
            try (Attendee attendeeSession = new Attendee(email, password)) {
                attendeeSession.login();
                break;
            } catch (BMSException e) {
                View.displayError(e.getMessage());
            }
        }
    }

    private void loginAdmin() throws SQLException{
        while (true) {
            String email = getStr("[Admin] Email");
            char[] password = getPasswd("[Admin] Password");
            try (Admin adminSession = new Admin(email, password)) {
                adminSession.login();
                break;
            } catch (BMSException e) {
                View.displayError(e.getMessage());
            }
        }
    }

    private void registerAttendee() throws SQLException{
        Attr ID = AttendeeAttr.EMAIL;
        String[] finalVals = new String[9];
        finalVals[1] = ID.inputUniqueVal();
        finalVals[0] = ++attendeeNum + "";
        char[] rawPasswd = getNewPasswd("Password");
        finalVals[2] = SecurityUtils.toHash(rawPasswd);
        Attr[] infos = Arrays.copyOfRange(AttendeeAttr.values(), 3, 9);
        String[] infoVals = Attr.inputNewVals(infos);
        System.arraycopy(infoVals, 0, finalVals, 3, 6);

        try {
            Tables.ATTENDEE.insert(finalVals);
            View.displayMessage("Registration Successful");
        } catch (BMSException e) {
            View.displayError(e.getMessage());
            return;
        }

        // Directly login
        try (Attendee attendeeSession = new Attendee(finalVals[1], rawPasswd)) {
            attendeeSession.login();
        } catch (BMSException e) {
            View.displayError(e.getMessage());
        }
    }
}
