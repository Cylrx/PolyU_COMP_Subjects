package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidValueException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;
import static org.junit.Assert.*;

public class SizeCriterionTest {

    // Static Tests
    @Test
    public void testValidSizeCriterion() throws InvalidCriterionException, InvalidValueException, InvalidFileException, InvalidTypeException {
        SizeCriterion sc = new SizeCriterion("SizeCrt", ">", "100");
        assertEquals("SizeCrt", sc.getCrtName());
        assertEquals("(size > 100)", sc.getExpr());

        Document doc1 = new Document(null, "Doc1", repeat("x", 30), "txt"); // size = 40 + 60 = 100
        Document doc2 = new Document(null, "Doc2", repeat("z", 31), "java"); // size = 40 + 62 = 102
        Directory dir1 = new Directory("Dir1", null); // size = 40
        Directory dir2 = new Directory("Dir2", null); // size = 40

        assertFalse(sc.eval(doc1));
        assertTrue(sc.eval(doc2));
        assertFalse(sc.eval(dir1));
        assertFalse(sc.eval(dir2));
    }

    @Test
    public void testInvalidComparator() {
        try {
            new SizeCriterion("SizeCrt", "!==", "50");
            fail("Expected InvalidCriterionException due to invalid comparator");
        } catch (InvalidCriterionException | InvalidValueException e){
            assertEquals("invalid opcode: expects {<, >, <=, >=, !=, ==}, got !==", e.getMessage());
        }
    }

    @Test
    public void testInvalidOperand() {
        try {
            new SizeCriterion("SizeCrt", "<", "abc");
            fail("Expected InvalidCriterionException due to non-integer operand");
        } catch (InvalidCriterionException e){
            assertEquals("invalid operand: expects integer, got abc", e.getMessage());
        } catch (InvalidValueException e)  {
            fail("Unexpected InvalidValueException: " + e.getMessage());
        }
    }

    // Fuzz Tests
    @Test
    public void fuzzTestSizeCriterionEval() throws InvalidCriterionException, InvalidValueException, InvalidFileException, InvalidTypeException {
        Random rand = new Random();
        Directory dummyPDir = new Directory("pDir", null);
        String[] comparators = {"<", ">", "<=", ">=", "==", "!="};
        int T = 500, dirTCnt = 0, docTCnt = 0;
        for (int t = 0; t < T; t++) {
            String cmp = comparators[rand.nextInt(comparators.length)];
            int target = rand.nextInt(500) + 1;

            SizeCriterion sc = new SizeCriterion("sc", cmp, String.valueOf(target));

            for (int i = 0; i < T; i++) {
                int contentLength = rand.nextInt(10);
                long expectedSize = 40 + ((long) contentLength * 2);

                Document doc = new Document(null, "Doc" + i, getRandStr(rand, contentLength), "txt");
                Directory dir = new Directory("Dir" + i, dummyPDir); // size = 40

                boolean expectedDoc = compare(expectedSize, cmp, target);
                boolean expectedDir = compare(40, cmp, target);
                if (expectedDir) dirTCnt++;
                if (expectedDoc) docTCnt++;

                assertEquals(expectedDoc, sc.eval(doc));
                assertEquals(expectedDir, sc.eval(dir));
            }
        }

        System.out.printf("%d directory matches out of %d trials\n", dirTCnt, T * T);
        System.out.printf("%d document matches out of %d trials\n", docTCnt, T * T);
    }

    @Test public void fuzzTestSizeCriterionExpr() throws InvalidValueException, InvalidCriterionException {
        Random rand = new Random();
        String[] comparators = {"<", ">", "<=", ">=", "==", "!="};

        int T = 100;
        for (int t = 0; t < T; t++) {
            String cmp = comparators[rand.nextInt(comparators.length)];
            long target = rand.nextInt(500) + 1;
            SizeCriterion sc = new SizeCriterion("sc" + t, cmp, String.valueOf(target));
            String expect = String.format("(size %s %d)", cmp, target);
            assertEquals(expect, sc.getExpr());
        }
    }
}