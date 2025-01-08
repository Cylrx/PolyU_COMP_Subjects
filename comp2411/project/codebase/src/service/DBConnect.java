package service;

import java.sql.*;

//import oracle.jdbc.driver.*;
import org.sqlite.*;
import view.View;

import javax.sql.rowset.CachedRowSet;
import javax.sql.rowset.RowSetProvider;

public class DBConnect {
    //private final static String username = "system";
    //private final static String passwd = "oracledb";
    private static SQLiteConnection connection = null;

    public DBConnect(String url) {
        try {
            connection = (SQLiteConnection) DriverManager.getConnection(url);
            System.out.println("Connected.");
        } catch (SQLException e) {
            View.displayError("Database connection failed to establish");
            throw new RuntimeException(e);
        }
    }

    public void close() throws SQLException {
        try {
            connection.close();
            System.out.println("Connection closed");
        } catch (SQLException e) {
            View.displayError("Database connection failed to close");
            throw e;
        }
    }

    /**
     * for UPDATE, DELETE, INSERT
     */
    public void executeUpdate(String sql, String[] params) throws SQLException{
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            for (int i = 0; i < params.length; i++) {
                pstmt.setObject(i + 1, params[i]);
            }
            pstmt.execute();
        } catch (SQLException e) {
            View.displayError("Failed to execute: " + sql);
            throw e;
        }
    }

    public ResultSet executeQuery(String sql, Object... params) throws SQLException{
        try(PreparedStatement pstmt = connection.prepareStatement(sql)){
            for (int i = 0; i < params.length; i++)
                pstmt.setObject(i + 1, params[i]);

            // copies ResultSet to memory for offline manipulation
            // solves the issue where ResultSet is closed when statement is closed by try-with-resources
            CachedRowSet crs = RowSetProvider.newFactory().createCachedRowSet();
            crs.populate(pstmt.executeQuery());
            return crs;
        } catch (SQLException e) {
            View.displayError("Failed to execute: " + sql);
            throw e;
        }
    }
}
