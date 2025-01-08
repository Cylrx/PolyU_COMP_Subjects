package hk.edu.polyu.comp.comp2021.assignment2.employee;

/**
 * Levels of salary.
 */
public enum SalaryLevel {
    ENTRY(1), JUNIOR(1.25), SENIOR(1.5), EXECUTIVE(2);

    // Task 1.5: Add missing code here.
    private double level;

    SalaryLevel(double level) {
        this.level = level;
    }

    public double getScale() {
        return level;
    }

}