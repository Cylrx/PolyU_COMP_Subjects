package service;

import exceptions.BMSException;

import java.sql.ResultSet;
import java.sql.SQLException;

import static controller.Controller.db;

public enum Tables {
    ATTENDEE("Attendee", AttendeeAttr.class),
    ADMIN("Administrator", AdminAttr.class),
    BANQUET("Banquet", BanquetAttr.class),
    MEAL("Meal", MealAttr.class),
    REGISTRATION("Registration", RegistrationAttr.class);

    private final String tableName;
    private final Class<? extends Attr> tableAttr;

    Tables(String tableName, Class<? extends Attr> tableAttr) {
        this.tableName = tableName;
        this.tableAttr = tableAttr;
    }

    /**
     * Query <code>columns</code> from <code>this</code> table based on <code>conditions</code>
     * i.e., SELECT <code>columns</code> FROM <code>this</code> WHERE <code>conditions</code>
     *
     * @param columns         the columns we wish to query
     * @param conditionClause the String condition in the format of (? = ? AND ? = ? ...)
     * @param conditionVals the values to fill in the condition clause
     * @return returns a <code>ResultSet</code> of the query results
     * @throws SQLException if an unexpected SQL exception occurred
     */
    public ResultSet query(String[] columns, String conditionClause, String[] conditionVals) throws SQLException {
        StringBuilder sql = new StringBuilder();
        // else sql = "SELECT" + " ?".repeat(columns.length) + " FROM " + tableName;

        if (columns.length == 0) sql.append("SELECT *");
        else {
            sql.append("SELECT ");
            for (int i = 0; i < columns.length; i++) {
                sql.append(columns[i]);
                if (!columns[i].equals("COUNT(*)"))
                    sql.append(" AS ").append(columns[i]);
                if (i < columns.length - 1) sql.append(", ");
            }
        }
        sql.append(" FROM ").append(tableName);
        if (!conditionClause.isEmpty()) sql.append(" WHERE ").append(conditionClause);
        return db.executeQuery(sql.toString(), (Object[]) conditionVals);
    }

    /**
     * Deletes entries from <code>this</code> table based on <code>conditions</code>
     *
     * @param conditionClause the String condition in the format of (? = ? AND ? = ? ...)
     * @param conditionVals the values to fill in the conditionClause
     * @throws SQLException when unexpected SQL exception occurred
     */
    public void delete(String conditionClause, String[] conditionVals) throws SQLException {
        String sql = "DELETE FROM " + tableName;
        if (!conditionClause.isEmpty()) sql += " WHERE " + conditionClause;
        db.executeUpdate(sql, conditionVals);
    }

    /**
     * Insert a complete entry into the table, filling every column.
     *
     * @param vals the complete tuple of values we wish to insert
     * @throws SQLException when unexpected SQL error occurred
     * @throws BMSException when the vals tuple already exist (violating uniqueness constraint)
     */
    public void insert(String... vals) throws SQLException, BMSException {
        if (hasEntry(vals)) throw new BMSException("Entry already exists");
        StringBuilder sql = new StringBuilder("INSERT INTO ");
        sql.append(tableName);
        Attr[] attrs = tableAttr.getEnumConstants();
        sql.append(" (");
        for (int i = 0; i < attrs.length; i++) {
            sql.append(attrs[i].getAttrName());
            if (i < attrs.length - 1) sql.append(", ");
        }
        sql.append(") VALUES (");
        for (int i = 0; i < vals.length; i++) {
            sql.append("?");
            if (i < vals.length - 1) sql.append(", ");
        }
        sql.append(")");
        db.executeUpdate(sql.toString(), vals);
    }


    private boolean hasEntry(String... vals) throws SQLException {
        StringBuilder conditions = new StringBuilder();
        Attr[] attrs = tableAttr.getEnumConstants();
        for (int i = 0; i < vals.length; i++) {
            conditions.append(attrs[i].getAttrName()).append(" = ?");
            if (i < vals.length - 1) conditions.append(" AND ");
        }
        try (ResultSet rs = query(new String[]{}, conditions.toString(), vals)) {
            return rs.next();
        }
    }

    @Override
    public String toString() {
        return tableName;
    }
}
