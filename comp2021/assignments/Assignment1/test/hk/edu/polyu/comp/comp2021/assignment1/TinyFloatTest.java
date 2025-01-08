package hk.edu.polyu.comp.comp2021.assignment1;

import org.junit.Test;
import static org.junit.Assert.*;

public class TinyFloatTest {
    public static float DELTA = 1E-6f;

    @Test
    public void test1(){
        assert(Math.abs(TinyFloat.fromString("00000000") - 1) < DELTA);
    }

    @Test
    public void test2(){
        assert(Math.abs(TinyFloat.fromString("00001000") - 2) < DELTA);
    }

}
