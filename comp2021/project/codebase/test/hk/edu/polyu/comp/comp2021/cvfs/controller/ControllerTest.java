package hk.edu.polyu.comp.comp2021.cvfs.controller;

import hk.edu.polyu.comp.comp2021.cvfs.view.View;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class ControllerTest {
    private Controller ct;

    @Before
    public void setUp() throws Exception {
        View dummyView = new View();
        ct = new Controller(dummyView);
    }

    @Test
    public void runInstruction() {

    }
}