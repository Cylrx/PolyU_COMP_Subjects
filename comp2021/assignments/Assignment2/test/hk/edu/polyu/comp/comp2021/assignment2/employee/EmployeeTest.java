package hk.edu.polyu.comp.comp2021.assignment2.employee;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class EmployeeTest {

    public static final double DELTA = 1E-6;

    @Test
    public void testSalaryLevel01() {
        assertEquals(SalaryLevel.ENTRY.getScale(), 1, DELTA);
    }

    @Test
    public void testEmployee01(){
        Employee employee1 = new Employee("A", SalaryLevel.ENTRY);
        assertEquals(employee1.salary(), 2000, DELTA);
    }

    @Test
    public void testManager01(){
        Manager manager1 = new Manager("A", SalaryLevel.EXECUTIVE, 0.5);
        assertEquals(manager1.salary(), 6000, EmployeeTest.DELTA);
    }

}
