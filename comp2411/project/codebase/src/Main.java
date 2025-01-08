import controller.Controller;

import java.sql.SQLException;

public class Main {
    public static void main(String[] args) {
        Controller controller = new Controller();
        try {
            controller.init("jdbc:sqlite:test_file1.db");
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}
