package controller;

import service.Attr;
import service.Tables;
import exceptions.BMSException;
import utils.SecurityUtils;

import java.sql.ResultSet;
import java.sql.SQLException;

interface User extends AutoCloseable {
    void login() throws SQLException, BMSException;

    static boolean auth(Attr ID, Attr PW, int id, char[] pw, boolean isAdmin) throws SQLException {
        String[] columns = new String[]{PW.getAttrName()};
        String conditionClause = ID.getAttrName() + " = ?";
        String[] conditionVals = new String[]{Integer.toString(id)};
        Tables T = (isAdmin) ? Tables.ADMIN : Tables.ATTENDEE;
        try (ResultSet rs = T.query(columns, conditionClause, conditionVals)){
            if (!rs.next()) return false;
            String hash = rs.getString(1);
            return SecurityUtils.checkPasswd(pw, hash);
        }
    }
}
