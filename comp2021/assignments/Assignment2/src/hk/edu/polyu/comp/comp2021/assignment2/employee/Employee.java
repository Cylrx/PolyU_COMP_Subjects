package hk.edu.polyu.comp.comp2021.assignment2.employee;

/**
 * An employee in a company.
 */
public class Employee{
    /**
     * Name of the employee.
     */
    private final String name;

    /**
     * Level of salary of the employee.
     */
    private SalaryLevel salaryLevel;

    /**
     * Return the name of the employee.
     */
    public String getName(){
        return name;
    }

    /**
     * Return the salary level of the employee.
     */
    public SalaryLevel getSalaryLevel(){
        return salaryLevel;
    }

    /**
     * Set the salary level.
     */
    public void setSalaryLevel(SalaryLevel salaryLevel){
        this.salaryLevel = salaryLevel;
    }

    /**
     * Initialize an employee object.
     */
    public Employee(String name, SalaryLevel level){
        // Task 1.1: Add missing code here.
        this.name = name;
        this.salaryLevel = level;
    }

    /**
     * Return the salary of the employee.
     */
    public double salary(){
        // The salary of an employee is computed as the multiplication
        // of the base salary (2000.0) and the scale of the employee's salary level.
        // Task 1.2: Add missing code here.
        return BASE_SALARY * salaryLevel.getScale();
    }

    /**
     * Base salary of all employees.
     */
    public static final double BASE_SALARY = 2000.0;

}
